import pypandoc

from pathlib import Path

docs_dir = Path('/Users/danila/Study/thesis/llm/docs')

for docx_file in docs_dir.glob('*.docx'):
    txt_file = docx_file.with_suffix('.txt')
    output = pypandoc.convert_file(
        str(docx_file), 'plain', outputfile=str(txt_file)
    )
