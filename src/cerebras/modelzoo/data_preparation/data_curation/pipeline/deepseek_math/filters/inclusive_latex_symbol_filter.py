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

import re
from urllib.parse import urlparse

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter


class InclusiveLatexSymbolFilter(BaseFilter):
    """
    Enhanced filter for detecting mathematical content with domain-aware classification..

    Pipeline:
    1. Basic Filter Compatibility Check (highest priority) - FIXED
    2. Domain Classification (Fast Filter)
    3. Domain-Specific Content Analysis

    Always returns True (passes all documents through) but sets metadata:
    - 'contains_latex_symbols': 1 if math content found, 0 otherwise
    - 'domain_classification': math/reject/academic/unknown
    - 'math_score': numerical confidence score
    """

    name = "InclusiveLatexSymbolFilter"

    # Comprehensive math-focused domains (auto-accept with content verification)
    MATH_DOMAINS = {
        # Academic repositories & preprint servers
        'arxiv.org',
        'hal.archives-ouvertes.fr',
        'mathscinet.ams.org',
        'zbmath.org',
        'projecteuclid.org',
        'researchgate.net',
        'academia.edu',
        # Mathematical societies & organizations
        'ams.org',
        'maa.org',
        'siam.org',
        'ims-org.org',
        'mathunion.org',
        'ems-ph.org',
        'cms.math.ca',
        'dmv.mathematik.de',
        'ams.org.au',
        'rsm.org.uk',
        'smf.emath.fr',
        'umi.dm.it',
        'real-sociedad-matematica.es',
        # Online math communities & Q&A
        'mathoverflow.net',
        'math.stackexchange.com',
        'mathscholar.org',
        'planetmath.org',
        'cut-the-knot.org',
        'artofproblemsolving.com',
        'brilliant.org',
        'mathpages.com',
        # Math reference & tools
        'mathworld.wolfram.com',
        'oeis.org',
        'wolframalpha.com',
        'geogebra.org',
        'desmos.com',
        'sage.org',
        'macaulay2.com',
        'gap-system.org',
        'magma.maths.usyd.edu.au',
        # Educational math platforms
        'khanacademy.org',
        'coursera.org',
        'edx.org',
        'udacity.com',
        'mathway.com',
        'symbolab.com',
        'photomath.com',
        'cymath.com',
        'quickmath.com',
        'mathpapa.com',
        'algebrator.com',
        'tiger-algebra.com',
        # Math competition & olympiad sites
        'imo-official.org',
        'amc.maa.org',
        'mathcounts.org',
        'usamo.wordpress.com',
        'mathlinks.ro',
        'kalva.demon.co.uk',
        # International math institutions
        'ihes.fr',
        'mpim-bonn.mpg.de',
        'msri.org',
        'ictp.it',
        'cirm-math.fr',
        'birs.ca',
        'mfo.de',
        'newton.ac.uk',
        'math.ias.edu',
        'claymath.org',
        'fields.utoronto.ca',
        'crm.umontreal.ca',
        'pims.math.ca',
        # Educational platforms with significant math content
        'libretexts.org',
        'overleaf.com',
        'wikibooks.org',
        'encyclopediaofmath.org',
        # Research platforms
        'research.google',
        'ncatlab.org',
        # Missing computational math/scientific libraries
        'dealii.org',
        'doc.cgal.org',
        'mofem.eng.gla.ac.uk',
        'dlmf.nist.gov',
        'fenicsproject.org',
        'sdurobotics.gitlab.io',
        # Missing calculator and converter sites
        'easy-divided-by.com',
        'calc.com',
        'web2.0calc.com',
        'percentages.io',
        'hextobinary.com',
        'kodytools.com',
        # Missing math community and puzzle sites
        'aperiodical.com',
        'forums.randommath.com',
        # Missing statistical/data science platforms
        'communities.sas.com',
        'freestatistics.org',
        # Missing international math/academic domains
        'qsms.math.snu.ac.kr',
        'eccc.weizmann.ac.il',
        'tabrizu.ac.ir',
        # Japanese/international Maple/math sites
        'jp.maplesoft.com',
        'nl.mathworks.com',
        # Additional discovered domains
        'mathspace.co',
        'wizako.com',
        'optimization-online.org',
        'smartsolve.ai',
        'askfilo.com',
        'vedantu.com',
        'vedclass.com',
        'cbseschoolnotes.com',
        'maplesoft.com',
        'amurchem.com',
        'biturbo.io',
        'biologynotesonline.com',
        'eanshub.com',
        'fresherbell.com',
        'ques10.com',
        'uniontestprep.com',
    }

    # University math departments (pattern-based recognition)
    UNIVERSITY_MATH_PATTERNS = [
        'math.mit.edu',
        'math.harvard.edu',
        'math.stanford.edu',
        'math.princeton.edu',
        'math.berkeley.edu',
        'math.caltech.edu',
        'math.yale.edu',
        'math.columbia.edu',
        'math.uchicago.edu',
        'math.ucla.edu',
        'math.umich.edu',
        'math.wisc.edu',
        'math.washington.edu',
        'math.gatech.edu',
        'math.cmu.edu',
        'math.cornell.edu',
        'math.duke.edu',
        'math.brown.edu',
        'math.dartmouth.edu',
        'math.upenn.edu',
        'math.rutgers.edu',
        'math.nyu.edu',
        'math.bu.edu',
        'math.tufts.edu',
        'math.northeastern.edu',
        'math.umd.edu',
        'math.utexas.edu',
        'math.rice.edu',
        'math.northwestern.edu',
        'math.umn.edu',
        'math.ohiou.edu',
        'math.psu.edu',
        'math.illinois.edu',
        'math.purdue.edu',
        'math.indiana.edu',
        'math.msu.edu',
        'math.uiowa.edu',
        'math.ku.edu',
        'math.missouri.edu',
        'math.unl.edu',
        'math.okstate.edu',
        'math.unt.edu',
        'math.uh.edu',
        'math.ttu.edu',
        'math.arizona.edu',
        'math.asu.edu',
        'math.utah.edu',
        'math.colorado.edu',
        'math.oregonstate.edu',
        'math.ubc.ca',
        'math.utoronto.ca',
        'math.mcmaster.ca',
        'math.uwaterloo.ca',
        'math.queensu.ca',
        'math.ualberta.ca',
        'math.sfu.ca',
        'math.ox.ac.uk',
        'math.cam.ac.uk',
        'math.imperial.ac.uk',
        'math.warwick.ac.uk',
        'math.bristol.ac.uk',
        'math.bath.ac.uk',
        'math.edinburgh.ac.uk',
        'math.manchester.ac.uk',
        'math.nottingham.ac.uk',
        'math.leeds.ac.uk',
        'math.sheffield.ac.uk',
        'math.soton.ac.uk',
        'math.kcl.ac.uk',
        'math.qmul.ac.uk',
        'math.ucl.ac.uk',
        'dpmms.cam.ac.uk',
        'maths.ox.ac.uk',
        'mathematics.ox.ac.uk',
    ]

    # Never-math domains (auto-reject)
    NEVER_MATH_DOMAINS = {
        # News & Media
        'cnn.com',
        'bbc.com',
        'reuters.com',
        'ap.org',
        'npr.org',
        'abc.com',
        'cbs.com',
        'nbc.com',
        'fox.com',
        'msnbc.com',
        'cnbc.com',
        'bloomberg.com',
        'wsj.com',
        'nytimes.com',
        'washingtonpost.com',
        'usatoday.com',
        'latimes.com',
        'chicagotribune.com',
        'bostonglobe.com',
        'guardian.co.uk',
        'independent.co.uk',
        'telegraph.co.uk',
        'dailymail.co.uk',
        'thesun.co.uk',
        'mirror.co.uk',
        'express.co.uk',
        'metro.co.uk',
        'huffpost.com',
        'buzzfeed.com',
        'vox.com',
        'slate.com',
        'salon.com',
        'politico.com',
        'thehill.com',
        'axios.com',
        # Social Media
        'facebook.com',
        'instagram.com',
        'twitter.com',
        'linkedin.com',
        'tiktok.com',
        'snapchat.com',
        'pinterest.com',
        'reddit.com',
        'tumblr.com',
        'flickr.com',
        'whatsapp.com',
        'telegram.org',
        'discord.com',
        'skype.com',
        'zoom.us',
        'slack.com',
        'teams.microsoft.com',
        # E-commerce & Shopping
        'amazon.com',
        'ebay.com',
        'walmart.com',
        'target.com',
        'costco.com',
        'homedepot.com',
        'lowes.com',
        'bestbuy.com',
        'macys.com',
        'nordstrom.com',
        'kohls.com',
        'jcpenney.com',
        'sears.com',
        'overstock.com',
        'wayfair.com',
        'etsy.com',
        'shopify.com',
        'alibaba.com',
        'aliexpress.com',
        'wish.com',
        'mercari.com',
        'poshmark.com',
        'depop.com',
        'vinted.com',
        'thredup.com',
        # Entertainment & Streaming
        'youtube.com',
        'netflix.com',
        'hulu.com',
        'amazonprime.com',
        'disneyplus.com',
        'hbomax.com',
        'paramount.com',
        'peacocktv.com',
        'appletv.com',
        'crunchyroll.com',
        'funimation.com',
        'spotify.com',
        'applemusic.com',
        'pandora.com',
        'soundcloud.com',
        'deezer.com',
        'tidal.com',
        'last.fm',
        'bandcamp.com',
        # Gaming
        'steam.com',
        'epicgames.com',
        'origin.com',
        'uplay.com',
        'gog.com',
        'battle.net',
        'riotgames.com',
        'blizzard.com',
        'ea.com',
        'activision.com',
        'ubisoft.com',
        'nintendo.com',
        'sony.com',
        'microsoft.com',
        'xbox.com',
        'playstation.com',
        'twitch.tv',
        'mixer.com',
        # Sports
        'espn.com',
        'foxsports.com',
        'cbssports.com',
        'nbcsports.com',
        'si.com',
        'bleacherreport.com',
        'sbnation.com',
        'theringer.com',
        'athletic.com',
        'tsn.ca',
        'sportsnet.ca',
        'skysports.com',
        'bbc.co.uk/sport',
        'eurosport.com',
        'nfl.com',
        'nba.com',
        'mlb.com',
        'nhl.com',
        'mls.com',
        'fifa.com',
        'uefa.com',
        # Travel & Lifestyle
        'booking.com',
        'expedia.com',
        'priceline.com',
        'kayak.com',
        'tripadvisor.com',
        'airbnb.com',
        'vrbo.com',
        'hotels.com',
        'marriott.com',
        'hilton.com',
        'hyatt.com',
        'ihg.com',
        'accor.com',
        'trivago.com',
        'agoda.com',
        'hostelworld.com',
        # Food & Recipes
        'allrecipes.com',
        'foodnetwork.com',
        'epicurious.com',
        'tastingtable.com',
        'bonappetit.com',
        'foodandwine.com',
        'eater.com',
        'seriouseats.com',
        'delish.com',
        'yummly.com',
        'bigoven.com',
        'recipezaar.com',
        'cookpad.com',
        # Business & Finance (General)
        'forbes.com',
        'fortune.com',
        'businessinsider.com',
        'fastcompany.com',
        'inc.com',
        'entrepreneur.com',
        'marketwatch.com',
        'morningstar.com',
        'fool.com',
        'investopedia.com',
        'bankrate.com',
        'nerdwallet.com',
        'creditkarma.com',
        'mint.com',
        'personalcapital.com',
        'robinhood.com',
        'etrade.com',
        'schwab.com',
        'fidelity.com',
        'vanguard.com',
    }

    # Academic research repository patterns (domain + path combinations)
    ACADEMIC_RESEARCH_PATTERNS = [
        ('collaborate.princeton.edu', ['/publications/', '/en/publications/']),
        ('cris.haifa.ac.il', ['/publications/', '/en/publications/']),
        ('cris.openu.ac.il', ['/publications/', '/en/publications/']),
        ('pure.uai.cl', ['/publications/', '/es/publications/']),
        (
            'perfilesycapacidades.javeriana.edu.co',
            ['/publications/', '/es/publications/'],
        ),
        ('par.nsf.gov', ['/search/']),
        ('pubs.aip.org', ['/']),  # Physics/math journals
        ('synthical.com', ['/search/']),
        ('hepdata.net', ['/search/']),
    ]

    # Academic personal page patterns (domain + path combinations)
    ACADEMIC_PERSONAL_PATTERNS = [
        ('users.wfu.edu', ['/2adic/', '/math/', '/mathematics/']),
        ('math.bu.edu', ['/people/', '/publication/']),
        ('kundudeb.github.io', ['/']),  # Academic GitHub pages
        ('juanitorduz.github.io', ['/']),  # Data science/stats GitHub pages
    ]

    # Academic domains that might have math content (use content analysis)
    ACADEMIC_DOMAIN_TLDS = {
        '.edu',
        '.ac.uk',
        '.edu.au',
        '.ac.in',
        '.ac.jp',
        '.ac.kr',
        '.ac.cn',
        '.ac.il',
        '.edu.co',
        '.ac.nz',
        '.edu.sg',
        '.ac.za',
        '.edu.br',
    }

    # Academic keywords in domain names
    ACADEMIC_KEYWORDS = {'university', 'college', 'institute', 'school', 'edu'}

    # Math-related keywords in domains/URLs
    MATH_DOMAIN_KEYWORDS = {
        'math',
        'mathematics',
        'calculus',
        'algebra',
        'geometry',
        'statistics',
        'probability',
        'analysis',
        'topology',
        'number-theory',
        'combinatorics',
    }

    # Mathematical content path indicators
    MATH_PATH_INDICATORS = [
        '/math/',
        '/mathematics/',
        '/calculus/',
        '/algebra/',
        '/geometry/',
        '/statistics/',
        '/probability/',
        '/analysis/',
        '/topology/',
        '/2adic/',
        '/publications/',
        '/research/',
        '/paper/',
        '/theorem/',
        '/proof/',
        '/equation/',
        '/formula/',
        '/solver/',
        '/calculator/',
        '/conversion/',
        '/units/',
        '/optimization/',
        '/algorithm/',
        '/ml/',
        '/ai/',
        '/data-science/',
        '/quantitative/',
        '/numerical/',
    ]

    # FIXED: Math-specific LaTeX commands only (removed generic LaTeX formatting)
    MATH_SPECIFIC_COMMANDS = [
        # Mathematical operators and symbols
        "\\frac",
        "\\int",
        "\\sum",
        "\\prod",
        "\\lim",
        "\\partial",
        "\\nabla",
        "\\infty",
        "\\sqrt",
        "\\root",
        "\\binom",
        # Greek letters (clearly mathematical in most contexts)
        "\\alpha",
        "\\beta",
        "\\gamma",
        "\\delta",
        "\\epsilon",
        "\\zeta",
        "\\eta",
        "\\theta",
        "\\iota",
        "\\kappa",
        "\\lambda",
        "\\mu",
        "\\nu",
        "\\xi",
        "\\pi",
        "\\rho",
        "\\sigma",
        "\\tau",
        "\\phi",
        "\\chi",
        "\\psi",
        "\\omega",
        "\\Gamma",
        "\\Delta",
        "\\Theta",
        "\\Lambda",
        "\\Xi",
        "\\Pi",
        "\\Sigma",
        "\\Phi",
        "\\Psi",
        "\\Omega",
        "\\varepsilon",
        "\\varphi",
        # Mathematical relations and operators
        "\\leq",
        "\\geq",
        "\\neq",
        "\\approx",
        "\\equiv",
        "\\sim",
        "\\subset",
        "\\supset",
        "\\in",
        "\\notin",
        "\\cap",
        "\\cup",
        "\\wedge",
        "\\vee",
        "\\oplus",
        "\\otimes",
        "\\times",
        "\\cdot",
        "\\pm",
        "\\mp",
        "\\rightarrow",
        "\\leftarrow",
        "\\leftrightarrow",
        "\\Rightarrow",
        "\\Leftarrow",
        "\\Leftrightarrow",
        # Mathematical functions
        "\\sin",
        "\\cos",
        "\\tan",
        "\\cot",
        "\\sec",
        "\\csc",
        "\\arcsin",
        "\\arccos",
        "\\arctan",
        "\\log",
        "\\ln",
        "\\exp",
        "\\det",
        "\\dim",
        "\\ker",
        "\\deg",
        "\\gcd",
        "\\lcm",
        # Mathematical structures and formatting
        "\\mathbb",
        "\\mathcal",
        "\\mathfrak",
        "\\mathbf",
        "\\mathrm",
        "\\vec",
        "\\dot",
        "\\ddot",
        "\\hat",
        "\\tilde",
        "\\bar",
        "\\overline",
        "\\underline",
        "\\langle",
        "\\rangle",
        "\\over",  # Fraction notation \\over
        # Mathematical environments and theorem-like structures
        "\\theorem",
        "\\proof",
        "\\lemma",
        "\\corollary",
        "\\proposition",
        "\\definition",
        "\\example",
        "\\remark",
        "\\note",
        # Specific mathematical notation
        "\\ell",
        "\\prime",
        "\\forall",
        "\\exists",
        "\\emptyset",
        "\\setminus",
        "\\backslash",
    ]

    # Mathematical Unicode symbols (expanded)
    MATH_UNICODE_SYMBOLS = [
        # Unicode symbols
        "∑",
        "∫",
        "∞",
        "√",
        "∂",
        "∇",
        "∈",
        "∉",
        "⊂",
        "⊃",
        "∪",
        "∩",
        "∧",
        "∨",
        "∀",
        "∃",
        "∄",
        "∅",
        "⊕",
        "⊗",
        "≈",
        "≠",
        "≤",
        "≥",
        "≪",
        "≫",
        "∝",
        "∼",
        "≅",
        "≡",
        "α",
        "β",
        "γ",
        "δ",
        "ε",
        "ζ",
        "η",
        "θ",
        "ι",
        "κ",
        "λ",
        "μ",
        "ν",
        "ξ",
        "π",
        "ρ",
        "σ",
        "τ",
        "φ",
        "χ",
        "ψ",
        "ω",
        "Γ",
        "Δ",
        "Θ",
        "Λ",
        "Ξ",
        "Π",
        "Σ",
        "Φ",
        "Ψ",
        "Ω",
        "±",
        "×",
        "÷",
        "·",
        "∝",
        "⊥",
        "∥",
        "∠",
        "°",
        # ASCII representations
        "+-",
        ">=",
        "<=",
        "!=",
        "~=",
        "->",
        "=>",
        "<->",
        "|-",
        "sum_",
        "prod_",
        "int_",
        "lim_",
    ]

    # Plain text math patterns
    PLAIN_TEXT_MATH_PATTERNS = [
        r'\b[a-zA-Z]\s*=\s*[\d\+\-\*/\s]+',  # x = 5, y = 2x + 3
        r'\b[a-zA-Z]\([a-zA-Z]\)\s*=',  # f(x) =
        r'\b\d+\s*[\+\-\*/]\s*\d+\s*=',  # 2 + 2 =
        r'[\+\-\*/\^]\s*[a-zA-Z]\b',  # +x, -y, *z
        r'\b[a-zA-Z]\^[\d\w]',  # x^2, a^n
        r'\([^\)]{3,}\)\s*[\+\-\*/\^=]',  # (expression) +
        r'sqrt\([^\)]+\)',  # sqrt(x)
        r'\bsin\s*\(|\bcos\s*\(|\btan\s*\(|\blog\s*\(',  # trig functions
    ]

    # Mathematical context patterns
    CONTEXT_PATTERNS = [
        r'(?:Example|Problem|Exercise|Question)\s*\d*\s*:.*?[=\+\-\*/]',
        r'(?:Solution|Answer|Proof)\s*:',
        r'Step\s+\d+\s*:',
        r'Case\s+\d+\s*:',
        r'Let\s+[a-zA-Z]\s+(?:be|denote|represent)',
        r'(?:Find|Calculate|Compute|Solve|Evaluate|Prove|Show)\s+(?:that|the)',
    ]

    # Additional math rendering indicators
    STRONG_MATH_INDICATORS = [
        'MathJax',
        'mathjax',
        '<math',
        'math-container',
        'katex.min.css',
        'latex.php',
        'codecogs',
        'tex.cgi',
        '\\begin{equation}',
        '\\begin{align}',
        '\\begin{matrix}',
        '\\begin{array}',
        '\\begin{theorem}',
        '\\begin{proof}',
        '\\displaystyle',
        '\\mathbb{',
        '\\mathcal{',
        '\\mathfrak{',
        'math-inline',
        'math-display',
        'equation',
        'formula',
    ]

    def __init__(
        self,
        min_score=1,
        min_score_good_domains=0.5,
        require_multiple_signals=True,
        basic_filter_compatibility=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_score = min_score
        self.min_score_good_domains = min_score_good_domains
        self.require_multiple_signals = require_multiple_signals
        self.basic_filter_compatibility = basic_filter_compatibility

        # Compile regexes with improved command matching
        # Create a more precise regex for math commands that avoids substring matches
        math_cmd_pattern = (
            r'(?:'
            + '|'.join(
                re.escape(cmd) + r'(?![a-zA-Z])'
                for cmd in self.MATH_SPECIFIC_COMMANDS
            )
            + r')'
        )
        self.math_command_regex = re.compile(math_cmd_pattern)

        # FIXED: More specific patterns for mathematical environments
        # Even more strict: require complete environment pairs
        # Simplest approach - only match the most reliable math environments
        self.math_environment_regex = re.compile(
            r'\\begin\{'
            r'(?:equation|align|gather|multline|eqnarray|displaymath)'
            r'(?:\*)?'
            r'\}',
            re.IGNORECASE,
        )

        self.unicode_regex = re.compile(
            '|'.join(re.escape(x) for x in self.MATH_UNICODE_SYMBOLS)
        )
        self.inline_math_regex = re.compile(
            r'\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]'
        )
        self.equation_regex = re.compile(
            r'\\begin\{equation\}.*?\\end\{equation\}|\\begin\{align\}.*?\\end\{align\}',
            re.DOTALL,
        )

        # Additional regexes
        self.plain_math_regex = re.compile(
            '|'.join(self.PLAIN_TEXT_MATH_PATTERNS), re.IGNORECASE
        )
        self.context_regex = re.compile(
            '|'.join(self.CONTEXT_PATTERNS), re.IGNORECASE
        )
        self.strong_indicator_regex = re.compile(
            '|'.join(re.escape(x) for x in self.STRONG_MATH_INDICATORS),
            re.IGNORECASE,
        )

        # False positive detection patterns
        self.ecommerce_patterns = re.compile(
            r'add.to.cart|buy.now|price|shipping|product|sku|size|color|quantity|checkout|shop|store',
            re.IGNORECASE,
        )
        self.sports_patterns = re.compile(
            r'score|game|player|team|season|win|loss|batting|pitching|inning|quarter|half',
            re.IGNORECASE,
        )

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL, handling various URL formats."""
        if not url:
            return ""

        try:
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if domain.startswith('www.'):
                domain = domain[4:]

            return domain
        except:
            return ""

    def classify_domain(self, url: str) -> str:
        """Classify domain into categories: 'math', 'reject', 'academic', 'unknown'"""
        if not url:
            return 'unknown', ""

        domain = self.extract_domain(url)
        url_lower = url.lower()

        # Tier 1: Explicit math domains
        if (
            domain in self.MATH_DOMAINS
            or domain in self.UNIVERSITY_MATH_PATTERNS
        ):
            return 'math', domain

        # Check academic research repository patterns
        for repo_domain, paths in self.ACADEMIC_RESEARCH_PATTERNS:
            if domain == repo_domain:
                if any(path in url_lower for path in paths):
                    return 'math', domain
                return 'academic', domain

        # Check academic personal page patterns
        for personal_domain, paths in self.ACADEMIC_PERSONAL_PATTERNS:
            if domain == personal_domain:
                if any(path in url_lower for path in paths):
                    return 'math', domain
                return 'academic', domain

        # Tier 2: Never-math domains
        if domain in self.NEVER_MATH_DOMAINS:
            return 'reject', domain

        # Tier 3: Academic domains
        is_academic = False
        for tld in self.ACADEMIC_DOMAIN_TLDS:
            if domain.endswith(tld):
                is_academic = True
                break

        if not is_academic:
            for keyword in self.ACADEMIC_KEYWORDS:
                if keyword in domain:
                    is_academic = True
                    break

        if is_academic:
            if domain.startswith('math.') or domain.startswith('mathematics.'):
                return 'math', domain

            for keyword in self.MATH_DOMAIN_KEYWORDS:
                if keyword in domain or keyword in url_lower:
                    return 'math', domain

            if any(path in url_lower for path in self.MATH_PATH_INDICATORS):
                return 'math', domain

            return 'academic', domain

        # Tier 4: Special handling for mixed-content educational sites
        if 'libretexts.org' in domain:
            math_libretexts_indicators = [
                'math.libretexts.org',
                'stats.libretexts.org',
                'phys.libretexts.org',
                '/mathematics/',
                '/calculus/',
                '/algebra/',
                '/statistics/',
                '/probability/',
                '/geometry/',
                '/analysis/',
                '/topology/',
            ]
            if any(
                indicator in url_lower
                for indicator in math_libretexts_indicators
            ):
                return 'math', domain
            return 'academic', domain

        # Tier 5: General domains with math keywords
        for keyword in self.MATH_DOMAIN_KEYWORDS:
            if keyword in domain:
                return 'academic', domain

        # tool_keywords = ['calc', 'calculator', 'convert', 'tool', 'solver', 'math']
        # if any(keyword in domain for keyword in tool_keywords):
        #     return 'academic'

        # Tier 6: Fallback patterns for edge cases
        if domain.endswith('.github.io') or domain.endswith('.gitlab.io'):
            math_github_indicators = [
                '/math',
                '/statistics',
                '/data',
                '/ml',
                '/ai',
                '/algorithm',
            ]
            if any(
                indicator in url_lower for indicator in math_github_indicators
            ):
                return 'academic'

        # if domain.endswith('.appspot.com') or 'calc' in domain or 'convert' in domain:
        #     return 'academic'

        return 'unknown', domain

    def has_basic_math_content(self, text: str) -> bool:
        """Quick check for basic mathematical content (for math domain verification)."""
        # Check for math rendering systems
        if any(
            indicator in text.lower()
            for indicator in self.STRONG_MATH_INDICATORS
        ):
            return True

        # Check for LaTeX commands
        if len(self.math_command_regex.findall(text)) > 0:
            return True

        # Check for mathematical Unicode symbols
        if len(self.unicode_regex.findall(text)) > 0:
            return True

        # Check for inline math
        if len(self.inline_math_regex.findall(text)) > 0:
            return True

        # Check for plain text math patterns
        if len(self.plain_math_regex.findall(text)) > 0:
            return True

        # Check for mathematical context patterns
        if len(self.context_regex.findall(text)) > 0:
            return True

        # Check for mathematical expressions
        math_expressions = [
            r'\b[a-zA-Z]\s*=\s*[^=]{3,}',  # x = expression
            r'\b[a-zA-Z]\([a-zA-Z]\)\s*=',  # f(x) =
            r'\\[a-zA-Z]+\{',  # LaTeX commands with braces
            r'\$[^$]+\$',  # Inline math
            r'\\begin\{[^}]+\}',  # LaTeX environments
        ]

        for pattern in math_expressions:
            if re.search(pattern, text):
                return True

        # Check for mathematical terms
        math_context_terms = [
            'theorem',
            'proof',
            'lemma',
            'corollary',
            'proposition',
            'equation',
            'formula',
            'derivative',
            'integral',
            'matrix',
            'algorithm',
            'optimization',
            'convergence',
            'eigenvalue',
        ]

        text_lower = text.lower()
        math_term_count = sum(
            1 for term in math_context_terms if term in text_lower
        )
        if math_term_count >= 1:
            return True

        return False

    def is_non_math_page(self, url: str, text: str) -> bool:
        """Detect non-math pages even on math domains (about, careers, etc.)."""
        url_lower = url.lower()
        text_lower = text.lower()

        # Check URL patterns that suggest non-math content
        non_math_url_patterns = [
            '/about',
            '/contact',
            '/privacy',
            '/terms',
            '/careers',
            '/jobs',
            '/login',
            '/signup',
            '/account',
            '/profile',
            '/settings',
            '/blog/',
            '/news/',
            '/press/',
            '/media/',
            '/events/',
            '/home',
            '/index',
            '/sitemap',
            '/search',
        ]

        if any(pattern in url_lower for pattern in non_math_url_patterns):
            return True

        # Check content patterns
        non_math_content_patterns = [
            r'about\s+us|contact\s+us|privacy\s+policy|terms\s+of\s+service',
            r'job\s+listing|career|hiring|employment|apply\s+now',
            r'shopping\s+cart|checkout|my\s+account|login|sign\s+up',
            r'copyright|disclaimer|cookie\s+policy',
            r'navigation|site\s+map|home\s+page',
            r'404\s+error|page\s+not\s+found',
            r'coming\s+soon|under\s+construction',
        ]

        non_math_regex = re.compile(
            '|'.join(non_math_content_patterns), re.IGNORECASE
        )
        non_math_matches = len(non_math_regex.findall(text))

        return non_math_matches > 2

    def calculate_math_score(self, text: str, url: str = '') -> tuple:
        """Calculate a math score based on various indicators."""
        score = 0.0
        evidence = {}
        signal_types = 0

        text_lower = text.lower()

        # Check for strong negative signals first
        ecommerce_count = len(self.ecommerce_patterns.findall(text))
        sports_count = len(self.sports_patterns.findall(text))

        # Domain-based negative filtering
        if url:
            negative_domains = [
                'shop',
                'store',
                'buy',
                'product',
                'sports',
                'game',
                'mlb',
                'nba',
                'nfl',
            ]
            if any(domain in url.lower() for domain in negative_domains):
                score -= 0.3
                evidence['negative_domain'] = True

        # 1. Check rendering systems (strong signal)
        rendering_found = any(
            indicator in text_lower for indicator in self.STRONG_MATH_INDICATORS
        )
        if rendering_found:
            score += 0.5
            evidence['has_math_rendering'] = True
            signal_types += 1

        # 2. LaTeX commands (strong signal)
        latex_matches = len(self.math_command_regex.findall(text))
        if latex_matches >= 1:
            score += min(latex_matches * 0.1, 0.5)
            evidence['latex_commands'] = latex_matches
            signal_types += 1

        # 3. Plain text math patterns
        plain_math_matches = len(self.plain_math_regex.findall(text))
        if plain_math_matches >= 2:
            score += min(plain_math_matches * 0.05, 0.3)
            evidence['plain_math_patterns'] = plain_math_matches
            signal_types += 1

        # 4. Mathematical Unicode symbols
        symbol_count = len(self.unicode_regex.findall(text))
        if symbol_count >= 1:
            score += min(symbol_count * 0.02, 0.2)
            evidence['math_symbols'] = symbol_count
            signal_types += 1

        # 5. Inline/display math
        inline_math = len(self.inline_math_regex.findall(text))
        if inline_math >= 1:
            score += min(inline_math * 0.1, 0.3)
            evidence['inline_math'] = inline_math
            signal_types += 1

        # 6. Equation environments
        equations = len(self.equation_regex.findall(text))
        if equations >= 1:
            score += equations * 0.2
            evidence['equations'] = equations
            signal_types += 1

        # 7. Mathematical context patterns
        context_matches = len(self.context_regex.findall(text))
        if context_matches >= 1:
            score += min(context_matches * 0.15, 0.4)
            evidence['math_context'] = context_matches
            signal_types += 1

        # 8. Mathematical terms
        math_terms = [
            'equation',
            'formula',
            'function',
            'variable',
            'coefficient',
            'polynomial',
            'derivative',
            'integral',
            'matrix',
            'vector',
            'theorem',
            'proof',
            'lemma',
            'corollary',
            'proposition',
            'calculate',
            'compute',
            'solve',
            'evaluate',
            'simplify',
            'differentiate',
            'integrate',
            'factor',
            'expand',
            'given that',
            'such that',
            'for all',
            'there exists',
            'therefore',
            'thus',
            'hence',
            'QED',
            'if and only if',
            'perpendicular',
            'parallel',
            'angle',
            'triangle',
            'circle',
            'probability',
            'distribution',
            'mean',
            'variance',
            'standard deviation',
            'limit',
            'converge',
            'diverge',
            'sequence',
            'series',
            'domain',
            'range',
            'inverse',
            'composite',
            'one-to-one',
        ]

        math_term_count = sum(1 for term in math_terms if term in text_lower)
        if math_term_count >= 3:
            score += min(math_term_count * 0.03, 0.3)
            evidence['math_terms'] = math_term_count
            signal_types += 1

        # 9. Mathematical expressions and structure
        has_equations = bool(re.search(r'[a-zA-Z]\s*=\s*[^=]{3,}', text))
        has_proofs = bool(
            re.search(r'\b(?:proof|theorem|lemma|corollary)\b', text_lower)
        )
        has_math_sections = bool(
            re.search(r'(?:example|problem|exercise)\s*\d+', text_lower)
        )

        if has_equations:
            score += 0.2
            evidence['has_equations'] = True
            signal_types += 1
        if has_proofs:
            score += 0.3
            evidence['has_proofs'] = True
            signal_types += 1
        if has_math_sections:
            score += 0.2
            evidence['has_math_sections'] = True
            signal_types += 1

        # 10. Handle false positives with nuance
        if ecommerce_count > 5 or sports_count > 5:
            evidence['ecommerce_signals'] = ecommerce_count
            evidence['sports_signals'] = sports_count
            if rendering_found or latex_matches >= 10:
                score -= 0.2
            else:
                evidence['likely_false_positive'] = True
                score -= 0.5

        # 11. Require multiple signals only for unknown domains
        if self.require_multiple_signals and signal_types < 2:
            score *= 0.7
            evidence['single_signal_penalty'] = True

        # 12. Additional penalties for clear false positives
        date_pattern = r'\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        date_count = len(re.findall(date_pattern, text))
        if date_count > 10:
            score *= 0.8
            evidence['many_dates'] = date_count

        currency_pattern = r'[$€£¥]\s*[\d,]+\.?\d*'
        currency_count = len(re.findall(currency_pattern, text))
        if currency_count > 10:
            score *= 0.8
            evidence['many_currency'] = currency_count

        evidence['signal_types'] = signal_types

        return max(score, 0.0), evidence

    def filter(self, doc: Document) -> bool:
        """
        FIXED: More restrictive basic filter compatibility to reduce false positives.
        """
        text = doc.text
        url = doc.metadata.get('url', '')

        # Step 1: FIXED Basic filter compatibility check
        if self.basic_filter_compatibility:
            # Check original math-specific keywords (unchanged)
            basic_keywords = [
                'MathJax',
                'mathjax',
                '<math',
                'math-container',
                'katex.min.css',
                'latex.php',
                'codecogs',
                'tex.cgi',
                'class="tex"',
                "class='tex'",
            ]

            if any(keyword in text for keyword in basic_keywords):
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "basic_filter_keyword"
                return True

            # # FIXED: Check for mathematical environments first (more specific)
            if self.math_environment_regex.search(text):
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "math_environment"
                return True

            # FIXED: Only check for MATH-SPECIFIC LaTeX commands (with proper command boundary matching)
            math_specific_found = False
            for cmd in self.MATH_SPECIFIC_COMMANDS:
                # Use proper LaTeX command matching: command followed by non-letter or end of string
                # This prevents \\note from matching \\noindent
                pattern = re.escape(cmd) + r'(?![a-zA-Z])'
                if re.search(pattern, text):
                    math_specific_found = True
                    break

            if math_specific_found:
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "math_specific_latex"
                return True

            # FIXED: Additional check for inline math notation with mathematical content
            inline_math_pattern = r'\$[^$]*[\\α-ωΑ-Ω∑∫∞∂∇±≤≥≠∈∉⊂⊃∪∩][^$]*\$'
            if re.search(inline_math_pattern, text):
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "inline_math_with_symbols"
                return True

            # FIXED: Check for mathematical expressions (not just isolated commands)
            # Look for patterns like \frac{a}{b}, \sqrt{x}, etc.
            complex_math_pattern = r'\\(?:frac|sqrt|binom)\s*\{[^}]+\}'
            if re.search(complex_math_pattern, text):
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "complex_math_expression"
                return True

        # Steps 2-3: Continue with existing domain classification and content analysis
        domain_class, domain = self.classify_domain(url)
        doc.metadata["domain_classification"] = domain_class
        doc.metadata["domain"] = domain

        if domain_class == 'reject':
            doc.metadata["contains_latex_symbols"] = 0
            doc.metadata["math_score"] = 0.0
            doc.metadata["rejection_reason"] = "never_math_domain"
            return True

        elif domain_class == 'math':
            if self.is_non_math_page(url, text):
                doc.metadata["contains_latex_symbols"] = 0
                doc.metadata["math_score"] = 0.0
                doc.metadata["rejection_reason"] = (
                    "non_math_page_on_math_domain"
                )
                return True

            if self.has_basic_math_content(text):
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["math_score"] = 1.0
                doc.metadata["acceptance_reason"] = "math_domain_with_content"
                return True
            else:
                doc.metadata["contains_latex_symbols"] = 0
                doc.metadata["math_score"] = 0.2
                doc.metadata["rejection_reason"] = "math_domain_no_content"
                return True

        else:
            # Academic/unknown domains: full content analysis
            score, evidence = self.calculate_math_score(text, url)

            doc.metadata["math_score"] = score
            doc.metadata["math_evidence"] = evidence

            if domain_class == 'academic':
                threshold = self.min_score_good_domains
                if evidence.get('signal_types', 0) >= 1:
                    score += 0.1
            else:
                threshold = self.min_score
                if evidence.get('signal_types', 0) >= 2:
                    score += 0.05

            if score >= threshold:
                doc.metadata["contains_latex_symbols"] = 1
                doc.metadata["acceptance_reason"] = (
                    f"{domain_class}_domain_content_analysis"
                )
            else:
                doc.metadata["contains_latex_symbols"] = 0
                doc.metadata["rejection_reason"] = (
                    f"{domain_class}_domain_insufficient_content"
                )

            return True


def test_classification_cases():
    """Test cases for both false positives (should NOT be math) and true positives (should be math)"""

    false_positive_cases = [
        "\\begin{document}\n\\title{My CV}\n\\author{John Doe}\n\\end{document}",
        "Please see \\ref{section1} for more details. \\label{conclusion}",
        "\\textit{This is italicized text} and \\textbf{this is bold}.",
        "\\item First item \\item Second item \\item Third item",
        "\\hline separates table rows. \\vspace{10pt} adds vertical space.",
        "The results are shown in \\emph{Table 1}. See \\cite{author2023}.",
        "\\noindent This paragraph has no indentation. \\hspace{5mm} adds space.",
        "\\begin{figure}\\caption{A nice figure}\\end{figure}",
        "\\section{Introduction}\\subsection{Background}\\paragraph{Details}",
    ]

    true_positive_cases = [
        "The equation is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$",
        "\\begin{equation} E = mc^2 \\end{equation}",
        "Let $\\alpha$ and $\\beta$ be the roots of the polynomial.",
        "The integral $\\int_0^\\infty e^{-x} dx = 1$",
        "We have $\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$",
        "The function $f(x) = \\sin(x) + \\cos(x)$ has period $2\\pi$",
        "\\begin{align} x + y &= 5 \\\\ 2x - y &= 1 \\end{align}",
        "The probability is $P(A \\cap B) = P(A) \\cdot P(B|A)$",
        "Compute the derivative: $\\frac{d}{dx}[x^2] = 2x$",
        "The matrix $A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$",
    ]

    filter_instance = InclusiveLatexSymbolFilter()

    print(
        "=== Testing False Positive Cases (should NOT be detected as math) ==="
    )
    false_positive_passed = 0
    for i, case in enumerate(false_positive_cases):
        # Fixed: Provide required 'id' parameter for Document
        doc = Document(text=case, id=f"test_false_positive_{i}")
        filter_instance.filter(doc)
        contains_math = doc.metadata.get("contains_latex_symbols", 0)
        passed = contains_math == 0
        if passed:
            false_positive_passed += 1
        print(
            f"Test {i+1}: {'PASS' if passed else 'FAIL'} - Math detected: {contains_math}"
        )
        if contains_math == 1:
            print(
                f"  Reason: {doc.metadata.get('acceptance_reason', 'unknown')}"
            )
            # Debug: Show which math commands were found
            if doc.metadata.get('acceptance_reason') == 'math_specific_latex':
                found_commands = []
                for cmd in filter_instance.MATH_SPECIFIC_COMMANDS:
                    pattern = re.escape(cmd) + r'(?![a-zA-Z])'
                    if re.search(pattern, case):
                        found_commands.append(cmd)
                if found_commands:
                    print(f"  Debug: Found math commands: {found_commands}")
        print(f"  Text: {case[:80]}...")
        print()

    print(
        f"False Positive Tests: {false_positive_passed}/{len(false_positive_cases)} passed"
    )
    print()

    print("=== Testing True Positive Cases (SHOULD be detected as math) ===")
    true_positive_passed = 0
    for i, case in enumerate(true_positive_cases):
        # Fixed: Provide required 'id' parameter for Document
        doc = Document(text=case, id=f"test_true_positive_{i}")
        filter_instance.filter(doc)
        contains_math = doc.metadata.get("contains_latex_symbols", 0)
        passed = contains_math == 1
        if passed:
            true_positive_passed += 1
        print(
            f"Test {i+1}: {'PASS' if passed else 'FAIL'} - Math detected: {contains_math}"
        )
        if contains_math == 1:
            print(
                f"  Reason: {doc.metadata.get('acceptance_reason', 'unknown')}"
            )
        else:
            print(
                f"  Rejection reason: {doc.metadata.get('rejection_reason', 'unknown')}"
            )
        print(f"  Text: {case[:80]}...")
        print()

    print(
        f"True Positive Tests: {true_positive_passed}/{len(true_positive_cases)} passed"
    )
    print()
    print(
        f"Overall: {false_positive_passed + true_positive_passed}/{len(false_positive_cases) + len(true_positive_cases)} tests passed"
    )


if __name__ == "__main__":
    test_classification_cases()
