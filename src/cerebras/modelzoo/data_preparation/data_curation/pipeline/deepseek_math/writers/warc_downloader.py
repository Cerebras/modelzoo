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

import gzip
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

import requests
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from requests.adapters import HTTPAdapter
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter


class WarcDownloader(PipelineStep):
    """
    FIXED VERSION: A hybrid datatrove pipeline step with small thread pool for downloading URLs as WARC files.
    Balances simplicity with performance for distributed environments.
    """

    name = "WarcDownloader"

    def __init__(
        self,
        output_dir: str,
        max_warc_size: int = 1 * 1024**3,  # 1 GiB
        max_workers: int = 8,  # Small thread pool
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        checkpoint_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the WARC downloader.

        Args:
            output_dir: Directory to store WARC files
            max_warc_size: Maximum size per WARC file in bytes
            max_workers: Number of download threads (default: 8)
            timeout: Request timeout in seconds
            headers: HTTP headers for requests
            checkpoint_file: File to track processed URLs
        """
        super().__init__()

        self.output_dir_str = output_dir
        self.max_warc_size = max_warc_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.checkpoint_file = checkpoint_file

        # Store headers configuration
        self.headers_config = headers or {
            'User-Agent': 'parallel-warc-downloader/2.1',
            'Accept-Encoding': 'identity',
        }

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """
        Main pipeline run method that processes URL documents and downloads them.
        """
        # Setup runtime objects
        output_dir = Path(self.output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = self.checkpoint_file or str(
            output_dir / f"processed_urls_rank_{rank}.txt"
        )
        processed_urls = self._load_checkpoint(checkpoint_file)

        logger = logging.getLogger(f"{self.name}[{rank}]")
        logger.info(
            f"Starting WARC downloader for rank {rank}/{world_size} with {self.max_workers} workers"
        )

        # Initialize resources that need cleanup
        session = None
        warc_file = None
        warc_writer = None
        docs_processed = 0

        try:
            # Setup session with connection pooling
            session = requests.Session()
            session.headers.update(self.headers_config)
            adapter = HTTPAdapter(
                pool_connections=self.max_workers, pool_maxsize=self.max_workers
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # Thread-safe WARC writing
            warc_lock = Lock()
            warc_file, warc_writer = self._open_new_warc(output_dir, rank)
            warc_size = 0
            warc_count = 1

            urls_to_download = []

            # Collect URLs from documents
            for doc in data:
                self.stat_update('total')
                docs_processed += 1

                # Extract URL from document
                url = self._extract_url_from_document(doc)

                if url and url not in processed_urls:
                    urls_to_download.append(url)
                    self.stat_update('urls_to_download')
                elif url:
                    self.stat_update('urls_already_processed')
                else:
                    self.stat_update('urls_invalid')

                # Continue passing the document through the pipeline
                yield doc

            # Download URLs with thread pool
            if urls_to_download:
                logger.info(
                    f"Downloading {len(urls_to_download)} URLs as WARC files"
                )

                def download_and_write(url):
                    """Download a URL and write to WARC (thread-safe)."""
                    nonlocal warc_size, warc_count, warc_file, warc_writer

                    try:
                        # Download
                        resp = session.get(url, timeout=self.timeout)

                        # Prepare WARC record
                        headers = {
                            'status_code': resp.status_code,
                            'reason': resp.reason,
                        }
                        headers.update(resp.headers)
                        content = resp.content

                        # Remove problematic encoding headers
                        for h in [
                            'content-encoding',
                            'Content-Encoding',
                            'transfer-encoding',
                            'Transfer-Encoding',
                        ]:
                            headers.pop(h, None)

                        # Build HTTP headers for WARC
                        status = f"{headers.pop('status_code', 200)} {headers.pop('reason', 'OK')}"
                        http_header_list = [
                            (k, v)
                            for k, v in headers.items()
                            if k.lower().startswith('content-')
                        ]
                        http_headers = StatusAndHeaders(
                            status, http_header_list, protocol='HTTP/1.0'
                        )

                        # Thread-safe WARC writing
                        with warc_lock:
                            # Write WARC record
                            rec = warc_writer.create_warc_record(
                                uri=url,
                                record_type='response',
                                payload=BytesIO(content),
                                http_headers=http_headers,
                            )
                            warc_writer.write_record(rec)

                            # Update checkpoint
                            with open(
                                checkpoint_file, 'a', encoding='utf-8'
                            ) as f:
                                f.write(url + '\n')

                            warc_size += len(content)

                            # Check if we need to rotate WARC file
                            if warc_size >= self.max_warc_size:
                                warc_file.close()
                                warc_count += 1
                                warc_file, warc_writer = self._open_new_warc(
                                    output_dir, rank, warc_count
                                )
                                warc_size = 0
                                self.stat_update('warc_files_created')

                        self.stat_update('urls_downloaded')
                        return True

                    except Exception as e:
                        logger.warning(f"Failed to download {url}: {e}")
                        self.stat_update('urls_failed')
                        return False

                # Process URLs with thread pool
                completed = 0
                with ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    futures = {
                        executor.submit(download_and_write, url): url
                        for url in urls_to_download
                    }

                    for future in as_completed(futures):
                        completed += 1
                        if completed % 100 == 0:
                            logger.info(
                                f"Progress: {completed}/{len(urls_to_download)} URLs processed"
                            )

            logger.info(f"WARC downloader completed. Stats: {self.stats}")

        finally:
            # 🔧 CRITICAL FIX: Always cleanup resources
            if docs_processed > 0:
                print(
                    f"WarcDownloader processed {docs_processed} documents, cleaning up..."
                )

            # Close WARC file
            if warc_file:
                try:
                    warc_file.close()
                except Exception as e:
                    print(f" Error closing WARC file: {e}")

            # Close session
            if session:
                try:
                    session.close()
                except Exception as e:
                    print(f"Error closing session: {e}")

    def _load_checkpoint(self, checkpoint_file: str) -> set:
        """Load processed URLs from checkpoint file."""
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _extract_url_from_document(self, doc: Document) -> Optional[str]:
        """Extract URL from document text or metadata."""
        # First try metadata
        if doc.metadata and 'url' in doc.metadata:
            return doc.metadata['url']

        # Then try document text
        url = doc.text.strip()
        if url and (url.startswith('http://') or url.startswith('https://')):
            return url

        return None

    def _open_new_warc(self, output_dir: Path, rank: int, count: int = 1):
        """Open a new WARC file for writing."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = (
            output_dir / f"math_urls_rank_{rank:03d}_{ts}_{count:03d}.warc.gz"
        )
        warc_file = gzip.open(path, 'wb')
        warc_writer = WARCWriter(warc_file, gzip=False)
        return warc_file, warc_writer
