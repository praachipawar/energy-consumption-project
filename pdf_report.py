from weasyprint import HTML

def build_pdf(html_str: str, out_path: str):
    HTML(string=html_str).write_pdf(out_path)

def simple_html(summary: str) -> str:
    return f"""<html><body>
    <h2>Retrofit Plan</h2>
    <div style="white-space:pre-wrap">{summary}</div>
    </body></html>"""
