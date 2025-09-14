

def read_stdin(prompt: str) -> str:
    """Read a line from stdin."""
    return input(prompt)


def safe_print(piece: str):
    """Print safely, skip non-printable characters."""
    if piece is None or piece == "":
        return
    for c in piece:
        if c.isprintable() or c.isspace():
            print(c, end="")