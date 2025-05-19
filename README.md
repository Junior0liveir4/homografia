
# 🧠📸 Projeto de Homografia - Visão Computacional Aplicada

Este código implementa a costura (stitching) de imagens para formação de **panoramas** utilizando **homografia**, detecção de pontos-chave e blending com máscaras suavizadas. Ele foi desenvolvido como parte da disciplina **Visão Computacional Aplicada**.

---

## 🎯 Objetivo

Aplicar os conceitos de:
- Detecção de características visuais;
- Correspondência de pontos;
- Estimativa de homografia (transformações de perspectiva);
- Combinação de imagens (image blending);
- Montagem de panoramas com múltiplas imagens.

---

## 🔧 Funcionalidades

- ✅ Detecção de pontos SIFT e correspondência com FLANN;
- ✅ Estimativa robusta da homografia usando um método tipo RANSAC ("gold standard");
- ✅ Criação de máscaras com transições suaves (blending);
- ✅ Costura de múltiplas imagens sequenciais;
- ✅ Visualização de correspondências e resultado final;
- ✅ Salvamento automático do panorama resultante.

---

## 📂 Estrutura do Código

- `normaliza_pontos`, `meu_DLT`, `minha_homografia`: Etapas de estimativa de homografia normalizada;
- `gold_standard`: Método robusto para encontrar a melhor homografia com inliers;
- `match_detector`: Detecção e correspondência de pontos-chave com SIFT + FLANN;
- `registration`: Estima homografia entre duas imagens;
- `create_mask`: Cria máscaras com transições suaves para blending;
- `blending`: Alinha e mistura duas imagens com base na homografia e nas máscaras;
- `cria_panorama`: Função principal que costura uma lista de imagens sequencialmente.

---

## 🧰 Requisitos

Certifique-se de ter as seguintes bibliotecas Python instaladas:

```bash
pip install opencv-python numpy
```

Para ambientes com suporte a `SIFT`, utilize uma versão recente do OpenCV (≥4.4).

---

## 🚀 Como usar

1. **Prepare suas imagens** e coloque-as no mesmo diretório do script.
2. Atualize a lista `imgs_to_panorama` no final do código com os nomes corretos:
   ```python
   imgs_to_panorama = ['img1.jpg', 'img2.jpg', 'img3.jpg']
   ```
3. Execute o script:
   ```bash
   python3 homografia.py
   ```

O panorama será salvo automaticamente como `panorama_final.jpg`.

---

## ⚠️ Possíveis erros

- `cv2.imread(...) is None`: O caminho da imagem está incorreto ou a imagem não existe.
- Verifique se o diretório de execução do script contém as imagens.

---

## 🧪 Exemplo de uso

Dado um conjunto de imagens sobrepostas (como fotos panorâmicas tiradas lado a lado), o script:

1. Detecta e emparelha características visuais entre cada par;
2. Calula a transformação de perspectiva (homografia);
3. Costura as imagens de forma contínua;
4. Aplica blending com máscaras suavizadas para evitar cortes bruscos.

---

## 📘 Conceitos aplicados

- **Homografia**: transformação projetiva entre planos (ex: mudança de perspectiva entre imagens).
- **SIFT (Scale-Invariant Feature Transform)**: algoritmo robusto para detecção de pontos de interesse.
- **FLANN (Fast Library for Approximate Nearest Neighbors)**: busca eficiente por descritores similares.
- **Blending com máscaras**: suavização na transição entre imagens sobrepostas.

---

## 📚 Referências

- Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*
- OpenCV documentation: https://docs.opencv.org
- Aula de Visão Computacional Aplicada - LabSEA

---

## 👨‍🏫 Créditos

Desenvolvido como parte das atividades práticas da disciplina **Visão Computacional Aplicada** no **LabSEA**.

---

## 📬 Contato

Para dúvidas ou sugestões, entre em contato com o time do **LabSEA**.
