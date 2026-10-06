from fpdf import FPDF
import re

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', '', 12)

with open('Project_Documentation.md', 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = re.sub(r'[^\x00-\x7F]+', '', line) 
    
    if not line:
        pdf.ln(5)
        continue
    
    if line.startswith('# '):
        pdf.set_font('helvetica', 'B', 20)
        pdf.multi_cell(0, 10, line[2:].strip())
        pdf.ln(2)
    elif line.startswith('## '):
        pdf.set_font('helvetica', 'B', 16)
        pdf.multi_cell(0, 8, line[3:].strip())
        pdf.ln(2)
    elif line.startswith('### '):
        pdf.set_font('helvetica', 'B', 14)
        pdf.multi_cell(0, 8, line[4:].strip())
        pdf.ln(2)
    elif line == '---':
        pdf.ln(5)
    else:
        pdf.set_font('helvetica', '', 12)
        pdf.multi_cell(0, 6, line)
        pdf.ln(1)

pdf.output('Project_Documentation.pdf')
