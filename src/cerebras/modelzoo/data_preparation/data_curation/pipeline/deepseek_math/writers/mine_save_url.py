# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Dict, Iterator, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.jsonl import JsonlWriter


class Mine_Save_URL(PipelineStep):
    """
    Custom datatrove writer that mines URLs from sitemaps and saves them to separate JSONL.gz files
    Combines URLMiner functionality with JsonlWriter capabilities
    Writes domains to domain/filename and URLs to urls/filename
    """

    name = "Mine_Save_URL"

    def __init__(
        self,
        output_folder: str,
        compression: str = "gzip",
        # URLMiner parameters
        max_urls_per_sitemap: Optional[int] = None,
        delay: float = 1.0,
        headers: Dict[str, str] = None,
        timeout: int = 15,
        **kwargs,
    ):
        super().__init__()

        # Store only picklable configuration
        self.output_folder = output_folder
        self.compression = compression

        # URLMiner parameters (all picklable)
        self.max_urls_per_sitemap = max_urls_per_sitemap
        self.delay = delay
        self.headers_config = headers or {
            'User-Agent': 'Mozilla/5.0 (compatible; SitemapBot/1.0; +http://www.example.com/bot)'
        }
        self.timeout = timeout

        # These will be initialized in run() - None values are picklable ✅
        self._writers = None
        self._closed = False  # Add closed state tracking
        self.total_urls_saved = None
        self.domains_seen = None
        self.sitemaps_processed = None
        self.processed_domains = None
        self.domain_writer = None
        self.url_writer = None

    def _initialize_runtime_objects(self):
        """Initialize non-picklable objects at runtime"""
        # Initialize runtime state
        self._writers = {}
        self._closed = False
        self.total_urls_saved = 0
        self.domains_seen = set()
        self.sitemaps_processed = set()
        self.processed_domains = set()

    def _get_stat_value(self, key: str, default: int = 0) -> int:
        """Safely get a stat value with fallback"""
        try:
            return self.stats.__getitem__(key)
        except (KeyError, AttributeError):
            return default

    def _get_writer(self, subfolder: str, rank: int) -> JsonlWriter:
        """Return a JsonlWriter for the given subfolder, creating if needed."""
        if subfolder not in self._writers:
            output_folder = os.path.join(self.output_folder, subfolder)
            os.makedirs(output_folder, exist_ok=True)
            process_id = os.getpid()
            final_filename = f"mined_data_pid{process_id}_rank{rank}.jsonl.gz"
            writer = JsonlWriter(
                output_folder=output_folder,
                output_filename=final_filename,
            )
            self._writers[subfolder] = writer
        return self._writers[subfolder]

    def run(
        self, data: Iterator[Document], rank: int = 0, world_size: int = 1
    ) -> Iterator[Document]:
        """
        Main run method for datatrove pipeline.
        Processes documents and yields them back downstream.
        """
        # Initialize runtime objects on the worker
        self._initialize_runtime_objects()

        docs_processed = 0
        try:
            self.domain_writer = self._get_writer("domain", rank)
            self.url_writer = self._get_writer("urls", rank)

            for doc in data:
                try:
                    # print(f"Doc text = {doc.text}")
                    self._mine_and_save_from_document(doc, rank)

                    # Flush writers periodically to prevent data loss
                    if self.total_urls_saved % 100 == 0:
                        self._flush_writers()

                    docs_processed += 1
                    yield doc
                except Exception as e:
                    print(f"Error processing document {doc.id}: {e}")
                    self.stat_update('errors')
                    yield doc
        finally:
            # Close writers properly
            if docs_processed > 0:
                print(
                    f"Mine_Save_URL processed {docs_processed} documents, closing writers..."
                )
                self.close()

    def _flush_writers(self):
        """Flush all writers to ensure data is written to disk"""
        if self._writers:
            for writer in self._writers.values():
                if writer and hasattr(writer, 'flush'):
                    try:
                        writer.flush()
                    except Exception as e:
                        print(f"Warning: Failed to flush writer: {e}")

    def _mine_and_save_from_document(
        self, doc: Document, rank: int = 0, **kwargs
    ) -> None:
        """Mine URLs from domains in the document and save them"""
        # Extract domains from document
        domains = self._extract_domains_from_input(doc.text, **kwargs)

        # Filter out already processed domains
        new_domains = domains - self.processed_domains
        if not new_domains:
            return

        print(
            f"🗺️ Document {doc.id}: Mining URLs from {len(new_domains)} new domains..."
        )

        # Save domains to domain file
        for domain in new_domains:
            self._save_domain_document(domain, rank, **kwargs)

        # Discover sitemaps for domains
        domain_sitemaps = self._discover_sitemaps(new_domains)

        if not domain_sitemaps:
            print(f"❌ No sitemaps found for document {doc.id}")
            return

        print(f"✅ Found sitemaps for {len(domain_sitemaps)} domains")

        # Mine URLs from sitemaps and save them
        for domain, sitemap_url in domain_sitemaps.items():
            try:
                mined_urls = self._mine_single_sitemap(domain, sitemap_url)

                # Save each mined URL
                for url_info in mined_urls:
                    url_doc = Document(
                        text=url_info['url'],
                        id=f"mined_url_{self._get_stat_value('urls_extracted')}",
                        metadata={
                            'url': url_info['url'],
                            'domain': url_info['domain'],
                            'sitemap': url_info['sitemap'],
                        },
                    )

                    # Save the mined URL document
                    self._save_url_document(url_doc, rank, **kwargs)
                    self.stat_update('urls_extracted')

                # Mark domain as processed
                self.processed_domains.add(domain)

            except Exception as e:
                print(f"❌ Error mining sitemap for {domain}: {e}")
                self.stat_update('errors')

    def _save_domain_document(
        self, domain: str, rank: int = 0, **kwargs
    ) -> None:
        """Save a domain document using domain JsonlWriter"""
        try:
            domain_record = Document(
                text=domain,
                id=f"domain_{self._get_stat_value('domains_saved')}",
                metadata={'domain': domain, 'type': 'domain'},
            )

            # Update statistics
            self.stat_update('domains_saved')
            self.domains_seen.add(domain)

            # Use domain writer
            self.domain_writer.write(domain_record, rank, **kwargs)
        except Exception as e:
            print(f"❌ Error saving domain {domain}: {e}")
            self.stat_update('errors')

    def _save_url_document(
        self, doc: Document, rank: int = 0, **kwargs
    ) -> None:
        """Save a URL document using URL JsonlWriter"""
        try:
            # Create a new document with the URL data in the desired format
            url_record = Document(
                text=doc.text,  # Keep original URL as text
                id=doc.id,
                metadata={
                    'url': doc.text,
                    'domain': doc.metadata.get('domain', ''),
                    'sitemap': doc.metadata.get('sitemap', ''),
                    'type': 'url',
                },
            )

            # Update statistics
            self.total_urls_saved += 1
            if doc.metadata.get('sitemap'):
                self.sitemaps_processed.add(doc.metadata.get('sitemap'))

            # Progress reporting
            if self.total_urls_saved % 1000 == 0:
                print(
                    f"💾 Saved {self.total_urls_saved} URLs from {len(self.domains_seen)} domains..."
                )

            # Use URL writer
            self.url_writer.write(url_record, rank, **kwargs)
        except Exception as e:
            print(f"❌ Error saving URL {doc.text}: {e}")
            self.stat_update('errors')

    def _extract_domains_from_input(self, text: str, **kwargs) -> Set[str]:
        """Extract domain names from input text or metadata"""
        domains = set()
        # Try to extract domains from text (assume one domain per line)
        for line in text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                # Handle URLs or plain domains
                if line.startswith('http'):
                    domain = urlparse(line).netloc
                else:
                    domain = line

                if domain:
                    domains.add(domain)

        # Also check metadata for domains
        if 'domains' in kwargs:
            domains.update(kwargs['domains'])

        return domains

    def _discover_sitemaps(self, domains: Set[str]) -> Dict[str, str]:
        """Discover sitemaps for given domains"""
        domain_sitemaps = {}

        with requests.Session() as session:
            session.headers.update(self.headers_config)

            for i, domain in enumerate(domains, 1):
                try:
                    sitemap_url = self._find_sitemap(domain, session)

                    if sitemap_url:
                        domain_sitemaps[domain] = sitemap_url
                        print(f"  ✅ {domain} -> {sitemap_url}")
                        self.stat_update('sitemaps_found')
                    else:
                        print(f"  ❌ {domain} (no sitemap)")

                    self.stat_update('domains_processed')

                    if i % 10 == 0:
                        print(f"  Progress: {i}/{len(domains)} domains checked")

                    time.sleep(self.delay)

                except Exception as e:
                    print(f"  ❌ Error checking {domain}: {e}")
                    self.stat_update('errors')

        return domain_sitemaps

    def _find_sitemap(
        self, domain: str, session: requests.Session
    ) -> Optional[str]:
        """Find sitemap for a single domain"""
        # Common sitemap locations
        sitemap_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemaps.xml',
            '/sitemap.txt',
        ]

        base_url = f"https://{domain}"

        # First, try robots.txt
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            response = session.get(robots_url, timeout=self.timeout)

            if response.status_code == 200:
                for line in response.text.splitlines():
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        if self._is_valid_sitemap(sitemap_url, session):
                            return sitemap_url
        except:
            pass

        # Try common sitemap locations
        for path in sitemap_paths:
            try:
                sitemap_url = urljoin(base_url, path)
                if self._is_valid_sitemap(sitemap_url, session):
                    return sitemap_url
            except:
                continue

        return None

    def _is_valid_sitemap(self, url: str, session: requests.Session) -> bool:
        """Check if URL is a valid sitemap"""
        try:
            response = session.head(url, timeout=self.timeout)
            return response.status_code == 200
        except:
            return False

    def _mine_single_sitemap(self, domain: str, sitemap_url: str) -> List[Dict]:
        """IMPROVED: Pass max_urls directly to parser"""
        with requests.Session() as session:
            session.headers.update(self.headers_config)

            try:
                sitemap_urls = self._parse_sitemap(
                    sitemap_url, session, self.max_urls_per_sitemap
                )

                mined_urls = []
                for url in sitemap_urls:
                    mined_urls.append(
                        {
                            'url': url,
                            'domain': domain,
                            'sitemap': sitemap_url,
                            'discovered_via': 'sitemap',
                        }
                    )

                print(f"  {domain}: Found {len(mined_urls)} URLs")
                return mined_urls

            except Exception as e:
                print(f"  Failed to parse {sitemap_url}: {e}")
                return []

    def _parse_sitemap(
        self,
        sitemap_url: str,
        session: requests.Session,
        max_urls: Optional[int] = None,
    ) -> List[str]:
        """IMPROVED: Parse sitemap with early termination"""
        try:
            response = session.get(sitemap_url, timeout=self.timeout)
            if response.status_code != 200:
                return []

            # Parse XML sitemap
            if (
                b'<urlset' in response.content
                or b'<sitemapindex' in response.content
            ):
                soup = BeautifulSoup(response.content, 'xml')
                urls = []

                # Handle sitemap index
                if soup.find('sitemapindex'):
                    remaining_urls = max_urls
                    for i, loc_tag in enumerate(soup.find_all('loc')):
                        if i >= 10:  # Limit sub-sitemaps
                            break
                        if remaining_urls is not None and remaining_urls <= 0:
                            break

                        sub_sitemap_url = loc_tag.text.strip()
                        sub_urls = self._parse_sitemap(
                            sub_sitemap_url, session, remaining_urls
                        )
                        urls.extend(sub_urls)

                        if remaining_urls is not None:
                            remaining_urls -= len(sub_urls)

                # Handle regular sitemap - EARLY TERMINATION HERE
                else:
                    for i, loc_tag in enumerate(soup.find_all('loc')):
                        if max_urls is not None and i >= max_urls:
                            print(
                                f"    Reached limit of {max_urls} URLs, stopping parse"
                            )
                            break
                        urls.append(loc_tag.text.strip())

                return urls

            # Parse text sitemap
            elif response.headers.get('content-type', '').startswith(
                'text/plain'
            ):
                urls = []
                for i, line in enumerate(response.text.splitlines()):
                    if max_urls is not None and i >= max_urls:
                        break
                    line = line.strip()
                    if line and line.startswith('http'):
                        urls.append(line)
                return urls

        except Exception as e:
            print(f"    Error parsing {sitemap_url}: {e}")

        return []

    def close(self) -> None:
        """Close files and print final statistics"""

        if self._closed:
            return

        print("Closing writers and finalizing output...")

        # Close all writers safely with proper error handling
        if self._writers:
            for subfolder, writer in self._writers.items():
                if writer:
                    try:
                        # Flush before closing
                        if hasattr(writer, 'flush'):
                            writer.flush()

                        # Close the writer
                        writer.close()
                        print(f"Closed writer for {subfolder}")
                    except Exception as e:
                        print(f" Error closing writer for {subfolder}: {e}")

            # Clear writers dictionary
            self._writers.clear()

        self.domain_writer = None
        self.url_writer = None
        self._closed = True

        print(
            f"""
🎉 Final SaveURL Statistics:
• Domains processed: {self._get_stat_value('domains_processed')}
• Domains saved: {self._get_stat_value('domains_saved')}
• Sitemaps found: {self._get_stat_value('sitemaps_found')}
• URLs extracted: {self._get_stat_value('urls_extracted')}
• Total URLs saved: {self.total_urls_saved if self.total_urls_saved is not None else 0}
• Unique domains: {len(self.domains_seen) if self.domains_seen is not None else 0}
• Sitemaps processed: {len(self.sitemaps_processed) if self.sitemaps_processed is not None else 0}
• Errors: {self._get_stat_value('errors')}
• Output folder: {self.output_folder}
• Domain files: {self.output_folder}/domain/
• URL files: {self.output_folder}/urls/
        """
        )
