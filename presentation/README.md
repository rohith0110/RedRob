# Presentation Deliverables

Files:

- `redrob_candidate_ranking_system.pptx`
- `redrob_candidate_ranking_system.pdf`
- `assets/`: real sandbox screenshots and deck montage
- `build_deck.mjs`: artifact-tool source used to generate the PPTX

The deck only uses measured runtime values, actual release hashes and QA outcomes, and real screenshots from the local Streamlit sandbox.

PDF conversion command used in this environment:

```powershell
$pptPath = (Resolve-Path 'presentation/redrob_candidate_ranking_system.pptx').Path
$pdfPath = Join-Path (Split-Path $pptPath) 'redrob_candidate_ranking_system.pdf'
$powerpoint = New-Object -ComObject PowerPoint.Application
$powerpoint.Visible = -1
$presentation = $powerpoint.Presentations.Open($pptPath, $false, $false, $false)
$presentation.SaveAs($pdfPath, 32)
$presentation.Close()
$powerpoint.Quit()
```
