def convert_to_url(signature):
    """Converts a Pinterest image signature to a public URL."""
    prefix = 'https://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (
        signature[0:2],
        signature[2:4],
        signature[4:6],
        signature
    )