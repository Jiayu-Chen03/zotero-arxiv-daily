import tarfile
import re
import glob
import math
import smtplib
from collections import Counter
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from loguru import logger
import datetime
from omegaconf import DictConfig
import pymupdf
import pymupdf.layout
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.layout.activate()

import pymupdf4llm  # noqa: E402
from time import sleep

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_pick(query: str, candidates: dict[str, str], k1: float = 1.5, b: float = 0.75) -> str:
    """Return the candidate key whose content best matches *query* by BM25."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return next(iter(candidates))

    doc_tokens = {name: _tokenize(content) for name, content in candidates.items()}
    N = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens.values()) / max(N, 1)

    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(set(tokens))

    best_name, best_score = None, -1.0
    for name, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for q in query_tokens:
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            f_q = tf.get(q, 0)
            score += idf * (f_q * (k1 + 1)) / (f_q + k1 * (1 - b + b * dl / max(avgdl, 1)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def extract_tex_code_from_tar(file_path:str, paper_id:str, paper_title:str | None = None) -> dict[str,str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None
 
    tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None
    
    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file) :
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl','')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")

    file_contents = {}
    doc_block_candidates: list[str] = []
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8',errors='ignore')
        content = re.sub(r'%.*\n', '\n', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content) and not any(w in t for w in ['example', 'sample', 'template']):
            doc_block_candidates.append(t)
        file_contents[t] = content

    if main_tex is None:
        if len(doc_block_candidates) == 1:
            main_tex = doc_block_candidates[0]
            logger.debug(f"Choose {main_tex} as main tex file of {paper_id}")
        elif len(doc_block_candidates) > 1:
            if paper_title:
                main_tex = _bm25_pick(paper_title, {c: file_contents[c] for c in doc_block_candidates})
                logger.debug(f"Multiple document blocks found in {paper_id}; BM25 selected {main_tex} from {doc_block_candidates}")
            else:
                main_tex = doc_block_candidates[0]
                logger.debug(f"Multiple document blocks found in {paper_id}; no title provided, using first candidate {main_tex}")

    if main_tex is not None:
        main_source:str = file_contents[main_tex]
        #find and replace all included sub-files
        include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
        for f in include_files:
            if not f.endswith('.tex'):
                file_name = f + '.tex'
            else:
                file_name = f
            main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None
        
    tar.close()
    return file_contents

def extract_markdown_from_pdf(file_path:str) -> str:
    return pymupdf4llm.to_markdown(file_path,use_ocr=False,header=False,footer=False,ignore_code=True)

def glob_match(path:str, pattern:str) -> bool:
    re_pattern = glob.translate(pattern,recursive=True)
    return re.match(re_pattern, path) is not None

def send_email(config: DictConfig, html: str):
    sender = str(config.email.sender).strip()
    receiver = str(config.email.receiver).strip()
    password = str(config.email.sender_password).strip()
    smtp_server = str(config.email.smtp_server).strip()
    smtp_port = int(config.email.smtp_port)

    max_retries = 3
    retry_delay = 10
    timeout = 30

    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, "utf-8").encode(), addr))

    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = _format_addr(f"Github Action <{sender}>")
    msg["To"] = _format_addr(f"You <{receiver}>")

    today = datetime.datetime.now().strftime("%Y/%m/%d")
    msg["Subject"] = Header(f"Daily arXiv {today}", "utf-8").encode()

    last_exception = None

    connection_errors = (
        smtplib.SMTPException,
        ConnectionError,
        TimeoutError,
        OSError,
    )

    def _close(server):
        if server is not None:
            try:
                server.close()
            except Exception:
                pass

    def _connect():
        """Connect using implicit SSL, STARTTLS, or a plain SMTP fallback."""
        if smtp_port == 465:
            return smtplib.SMTP_SSL(
                smtp_server,
                smtp_port,
                timeout=timeout,
            )

        tls_server = None
        try:
            tls_server = smtplib.SMTP(
                smtp_server,
                smtp_port,
                timeout=timeout,
            )
            # starttls() performs EHLO when needed. Avoid requiring SMTP-like
            # implementations to expose ehlo() separately.
            tls_server.starttls()
            return tls_server
        except connection_errors as exc:
            _close(tls_server)
            logger.warning(
                f"STARTTLS connection failed: {type(exc).__name__}: {exc}; "
                "trying implicit SSL."
            )

        try:
            return smtplib.SMTP_SSL(
                smtp_server,
                smtp_port,
                timeout=timeout,
            )
        except connection_errors as exc:
            logger.warning(
                f"SSL connection failed: {type(exc).__name__}: {exc}; "
                "trying plain SMTP."
            )

        return smtplib.SMTP(
            smtp_server,
            smtp_port,
            timeout=timeout,
        )

    def _authenticate(server, attempt):
        """Try progressively more conservative SMTP AUTH handshakes."""
        if attempt == 1:
            logger.info("Authenticating with automatic SMTP AUTH negotiation...")
            return server.login(sender, password)

        if attempt == 2:
            logger.info("Authenticating without a SASL initial response...")
            return server.login(
                sender,
                password,
                initial_response_ok=False,
            )

        logger.info("Authenticating with explicit two-step AUTH LOGIN...")
        server.ehlo_or_helo_if_needed()
        advertised_auth = server.esmtp_features.get("auth", "").upper().split()
        if "LOGIN" not in advertised_auth:
            raise smtplib.SMTPNotSupportedError(
                "SMTP server does not advertise AUTH LOGIN"
            )

        responses = iter((sender, password))

        def _login_response(_challenge=None):
            return next(responses, "")

        return server.auth(
            "LOGIN",
            _login_response,
            initial_response_ok=False,
        )

    for attempt in range(1, max_retries + 1):
        server = None

        try:
            logger.info(
                f"Connecting to SMTP server "
                f"{smtp_server}:{smtp_port}, attempt {attempt}/{max_retries}"
            )

            server = _connect()

            logger.info("SMTP connected, logging in...")

            _authenticate(server, attempt)

            logger.info("SMTP login successful, sending email...")

            server.sendmail(
                sender,
                [receiver],
                msg.as_string(),
            )

            logger.info("Email sent successfully.")

            try:
                server.quit()
            except Exception:
                server.close()

            return

        except connection_errors as exc:
            last_exception = exc

            logger.warning(
                f"Email sending failed "
                f"(attempt {attempt}/{max_retries}): "
                f"{type(exc).__name__}: {exc}"
            )

            _close(server)

            if attempt < max_retries:
                sleep(retry_delay * attempt)

    provider_hint = ""
    if smtp_server.lower() == "smtp.qq.com":
        provider_hint = (
            " QQ Mail requires POP3/IMAP/SMTP to be enabled and "
            "SENDER_PASSWORD to contain a current 16-character authorization "
            "code, not the QQ account password."
        )

    raise RuntimeError(
        f"Failed to send email after {max_retries} attempts: "
        f"{type(last_exception).__name__}: {last_exception}.{provider_hint}"
    ) from last_exception
