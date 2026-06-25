from hashlib import sha256

from .models import ContentIdentity


def identify_content(
    kind: str,
    content: str,
) -> ContentIdentity:
    """
    Create a stable identity for delivered content.

    Line endings are normalized before hashing so equivalent output remains
    stable across operating systems.

    Args:
        kind: semantic content type, such as "symbol_source".
        content: delivered text.

    Returns:
        Stable content identity.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    encoded = normalized.encode("utf-8")
    digest = sha256(encoded).hexdigest()
    return ContentIdentity(
        content_id=f"{kind}:sha256:{digest}",
        kind=kind,
        digest=digest,
        byte_length=len(encoded),
    )
