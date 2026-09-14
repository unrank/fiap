"""
Monta o PDF de entrega da Fase 1.

Junta, em um unico documento:

    * capa com identificacao, link do repositorio Git e link do video;
    * o relatorio tecnico completo (``reports/RELATORIO_TECNICO.md``);
    * um anexo com todas as figuras geradas pelo pipeline, legendadas.

A conversao HTML -> PDF usa o Chrome ou o Edge em modo headless (nenhuma
dependencia extra de sistema). Se nenhum navegador for encontrado, o HTML
e gerado mesmo assim e basta abri-lo e usar "Imprimir -> Salvar como PDF".

Uso
---
    python scripts/build_pdf.py --repo https://github.com/usuario/repo \
                                --video https://youtu.be/XXXX \
                                --autores "Fulano, Beltrano"
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT_MD = ROOT / "reports" / "RELATORIO_TECNICO.md"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"
OUT_DIR = ROOT / "docs"

#: Legenda de cada figura no anexo, na ordem em que devem aparecer.
FIGURE_CAPTIONS: dict[str, str] = {
    "01_balanceamento_classes": "Distribuicao do alvo: 357 benignos (62,7%) e 212 malignos (37,3%).",
    "02_distribuicoes_por_classe": "Distribuicao das 9 variaveis mais discriminantes, separada por diagnostico.",
    "03_boxplots_por_classe": "Dispersao e outliers por diagnostico.",
    "04_matriz_correlacao": "Correlacao entre as 30 preditoras; os blocos vermelhos sao os grupos redundantes.",
    "05_correlacao_com_alvo": "Correlacao ponto-bisserial de cada variavel com o diagnostico.",
    "06_projecao_pca": "Projecao PCA: as classes ja se separam em duas dimensoes.",
    "07_dispersao_pareada": "Dispersao pareada das variaveis mais discriminantes.",
    "08_escala_e_assimetria": "Escalas e assimetrias que justificam a padronizacao e a imputacao pela mediana.",
    "09_comparacao_modelos": "Comparacao dos seis modelos no conjunto de validacao.",
    "10_curvas_roc_pr": "Curvas ROC e Precisao-Recall dos seis modelos (validacao).",
    "11_analise_limiar": "Calibracao do limiar: trade-off recall x precisao e o ponto escolhido.",
    "12_matriz_confusao_padrao": "Matriz de confusao no teste com o limiar padrao de 0,50.",
    "13_matriz_confusao_clinica": "Matriz de confusao no teste com o limiar clinico calibrado.",
    "14_distribuicao_scores": "Separacao das classes pela probabilidade prevista.",
    "15_calibracao": "Curva de calibracao: a probabilidade prevista corresponde ao risco observado.",
    "16_curva_aprendizado": "Curva de aprendizado do modelo campeao.",
    "17_importancia_nativa": "Coeficientes do modelo campeao (positivo empurra para maligno).",
    "18_referencia_logistica": "Modelo de referencia interpretavel: coeficientes da regressao logistica.",
    "19_importancia_permutacao": "Queda real de ROC AUC ao embaralhar cada variavel.",
    "20_shap_beeswarm": "SHAP beeswarm: magnitude e direcao do efeito de cada variavel.",
    "21_shap_importancia_global": "Importancia global SHAP (media do valor absoluto).",
    "22_3_shap_caso_verdadeiro_positivo": "Explicacao individual - caso corretamente sinalizado como maligno.",
    "22_2_shap_caso_falso_positivo": "Explicacao individual - falso positivo.",
    "22_1_shap_caso_falso_negativo": "Explicacao individual - falso negativo (o erro mais grave).",
    "22_4_shap_caso_caso_ambiguo": "Explicacao individual - caso ambiguo, proximo do limiar.",
    "30_amostras_radiografias": "EXTRA: exemplos do PneumoniaMNIST.",
    "31_cnn_curvas_treino": "EXTRA: evolucao da perda e da ROC AUC ao longo das epocas.",
    "32_cnn_matriz_confusao": "EXTRA: matriz de confusao da CNN no teste.",
    "33_cnn_predicoes": "EXTRA: predicoes da CNN (erros mais confiantes primeiro).",
    "34_cnn_gradcam": "EXTRA: Grad-CAM - onde a rede olhou para decidir.",
}

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1c2530; margin: 0;
}
h1, h2, h3, h4 { color: #0f2d3d; line-height: 1.25; }
h1 { font-size: 20pt; border-bottom: 3px solid #2E7D9A; padding-bottom: 6px;
     margin-top: 26px; page-break-before: always; }
h2 { font-size: 14.5pt; margin-top: 22px; border-bottom: 1px solid #d3dde3;
     padding-bottom: 4px; }
h3 { font-size: 12pt; margin-top: 16px; color: #17506b; }
h1, h2, h3, h4 { page-break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
a { color: #17506b; word-break: break-word; }
code { background: #eef3f6; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
pre { background: #f4f7f9; border-left: 3px solid #2E7D9A; padding: 9px 12px;
      overflow-x: auto; border-radius: 3px; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th { background: #2E7D9A; color: #fff; text-align: left; padding: 5px 7px;
     font-weight: 600; }
td { border-bottom: 1px solid #dde5ea; padding: 4px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f6f9fb; }
blockquote { border-left: 4px solid #EDAE49; background: #fdf7ec; margin: 12px 0;
             padding: 8px 14px; page-break-inside: avoid; }
blockquote p { margin: 4px 0; }
hr { border: 0; border-top: 1px solid #d3dde3; margin: 22px 0; }
ul, ol { margin: 7px 0 7px 18px; padding-left: 8px; }
li { margin: 3px 0; }

/* ---------------- capa ---------------- */
.capa { page-break-after: always; padding-top: 42mm; text-align: center; }
.capa .selo { font-size: 10pt; letter-spacing: 4px; color: #2E7D9A;
              text-transform: uppercase; font-weight: 700; }
.capa h1 { font-size: 30pt; border: 0; margin: 14px 0 4px; page-break-before: auto; }
.capa h2 { font-size: 15pt; border: 0; color: #41586a; font-weight: 400; margin: 0; }
.capa .regra { width: 90px; height: 4px; background: #D1495B; margin: 26px auto; }
.capa .links { margin-top: 34px; text-align: left; display: inline-block;
               background: #f4f7f9; border: 1px solid #dde5ea; border-radius: 6px;
               padding: 16px 22px; font-size: 10.5pt; min-width: 118mm; }
.capa .links div { margin: 7px 0; }
.capa .links b { color: #0f2d3d; }
.capa .links .rot { display: inline-block; min-width: 44mm; }
.capa .links .destaque { margin-top: 14px; padding-top: 12px;
                         border-top: 1px solid #dde5ea; }
.capa .rodape { margin-top: 40mm; font-size: 9pt; color: #6b7c8a; }
.capa .aviso { margin-top: 12mm; font-size: 11pt; font-weight: 600; color: #D1495B; }

/* ---------------- resumo ---------------- */
.resumo { background: #f4f7f9; border: 1px solid #dde5ea; border-radius: 6px;
          padding: 4px 18px 12px; margin: 16px 0; page-break-inside: avoid; }

/* ---------------- anexo de figuras ---------------- */
.figura { page-break-inside: avoid; margin: 16px 0 22px; text-align: center; }
.figura img { max-width: 100%; max-height: 205mm; border: 1px solid #dde5ea;
              border-radius: 4px; }
.figura .legenda { font-size: 9pt; color: #41586a; margin-top: 6px;
                   text-align: center; }
.figura .legenda b { color: #0f2d3d; }
"""


def _img_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_cover(repo: str, video: str, autores: str, resumo: dict) -> str:
    linhas = [
        ("Repositório Git", f'<a href="{html.escape(repo)}">{html.escape(repo)}</a>'),
        ("Vídeo (até 15 min)", f'<a href="{html.escape(video)}">{html.escape(video)}</a>'),
        ("Integrantes", html.escape(autores)),
        ("Data", date.today().strftime("%d/%m/%Y")),
    ]
    itens = "\n".join(
        f'<div><b class="rot">{k}:</b> {v}</div>' for k, v in linhas
    )

    campeao = resumo.get("teste", {}).get("campeao", "-")
    clinico = resumo.get("teste", {}).get("limiar_clinico", {})
    destaque = ""
    if clinico:
        destaque = (
            '<div class="destaque">'
            f'<b class="rot">Modelo campeão:</b> {html.escape(str(campeao))}</div>'
            f'<div><b class="rot">Resultado no teste:</b> '
            f"recall {clinico.get('recall', '-')} &middot; "
            f"F1 {clinico.get('f1', '-')} &middot; "
            f"ROC AUC {clinico.get('roc_auc', '-')}</div>"
        )

    return f"""
<section class="capa">
  <div class="selo">POS TECH &middot; Tech Challenge &middot; Fase 1</div>
  <h1>Sistema inteligente de<br>suporte ao diagnóstico</h1>
  <h2>Machine Learning aplicado à análise de exames médicos</h2>
  <div class="regra"></div>
  <div class="links">{itens}{destaque}</div>
  <div class="aviso">O modelo prioriza. O médico diagnostica.</div>
  <div class="rodape">
    Tarefa principal: diagnóstico de câncer de mama (Breast Cancer Wisconsin).<br>
    Entregável EXTRA: detecção de pneumonia em radiografias de tórax com CNN.
  </div>
</section>
"""


def build_figures_appendix() -> str:
    partes = ['<h1>Anexo A &mdash; Figuras geradas pelo pipeline</h1>',
              '<p>Todas as figuras abaixo sao produzidas automaticamente por '
              '<code>scripts/run_pipeline.py</code> e <code>scripts/run_cnn.py</code>, '
              'e ficam em <code>reports/figures/</code>.</p>']
    n = 0
    for nome, legenda in FIGURE_CAPTIONS.items():
        path = FIGURES_DIR / f"{nome}.png"
        if not path.exists():
            continue
        n += 1
        partes.append(
            f'<div class="figura">'
            f'<img src="{_img_data_uri(path)}" alt="{html.escape(nome)}">'
            f'<div class="legenda"><b>Figura {n}.</b> {html.escape(legenda)}'
            f' <em>({html.escape(nome)}.png)</em></div>'
            f"</div>"
        )
    print(f"anexo de figuras: {n} imagens embutidas")
    return "\n".join(partes)


def markdown_to_html(md_text: str) -> str:
    import markdown

    # A primeira linha (# titulo) ja aparece na capa; removemos para nao repetir.
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Imprime o HTML em PDF usando Chrome/Edge headless. Devolve True se deu certo."""
    candidatos = [
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    navegador = next((c for c in candidatos if c and Path(c).exists()), None)
    if navegador is None:
        print("nenhum Chrome/Edge encontrado - abra o HTML e use 'Imprimir > Salvar como PDF'")
        return False

    print(f"imprimindo com: {navegador}")
    cmd = [
        navegador, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not pdf_path.exists():
        print("falha ao gerar o PDF:", result.stderr[-600:])
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Gera o PDF de entrega da Fase 1")
    p.add_argument("--repo", default="https://github.com/SEU-USUARIO/SEU-REPOSITORIO")
    p.add_argument("--video", default="https://youtu.be/SEU-VIDEO")
    p.add_argument("--autores", default="(preencher com os integrantes do grupo)")
    p.add_argument("--nome", default="Tech_Challenge_Fase1")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resumo: dict = {}
    manifesto = METRICS_DIR / "00_manifesto_execucao.json"
    if manifesto.exists():
        resumo = json.loads(manifesto.read_text(encoding="utf-8"))

    md_text = REPORT_MD.read_text(encoding="utf-8")
    corpo = markdown_to_html(md_text)

    documento = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Tech Challenge Fase 1 - Sistema inteligente de suporte ao diagnostico</title>
<style>{CSS}</style>
</head>
<body>
{build_cover(args.repo, args.video, args.autores, resumo)}
{corpo}
{build_figures_appendix()}
</body>
</html>
"""

    html_path = OUT_DIR / f"{args.nome}.html"
    html_path.write_text(documento, encoding="utf-8")
    print(f"HTML gerado: {html_path} ({html_path.stat().st_size / 1024**2:.1f} MB)")

    pdf_path = OUT_DIR / f"{args.nome}.pdf"
    if html_to_pdf(html_path, pdf_path):
        print(f"PDF gerado: {pdf_path} ({pdf_path.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
