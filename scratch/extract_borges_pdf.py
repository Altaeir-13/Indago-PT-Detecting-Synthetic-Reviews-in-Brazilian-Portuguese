from pathlib import Path
import pdfplumber
p = Path(r'C:\Users\Altair\Downloads\Benchmarking_Machine_Learning_Algorithms_in_Fake_R.pdf')
out = Path(r'C:\Users\Altair\Documents\Working\Development\Indago\scratch\borges_extract.txt')
chunks = []
with pdfplumber.open(str(p)) as pdf:
    chunks.append(f'pages {len(pdf.pages)}\n')
    for i, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ''
        chunks.append(f'---PAGE {i}---\n{text}\n')
        tables = page.extract_tables() or []
        chunks.append(f'tables {len(tables)}\n')
        for ti, table in enumerate(tables, 1):
            chunks.append(f'---TABLE {i}.{ti}---\n')
            for row in table:
                chunks.append(repr(row) + '\n')
out.write_text(''.join(chunks), encoding='utf-8')
print(out)
