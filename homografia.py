import cv2
import numpy as np
import random

def normaliza_pontos(pontos):
    """
    Normaliza os pontos para estabilizar a estimativa da homografia.
    Calcula o centroide, a distância média e cria uma matriz de normalização.
    """
    # Calcula o centroide dos pontos
    centroide = np.mean(pontos, axis=1).reshape(-1, 1)
    # Subtrai o centroide para centralizar os pontos
    pontos_c = pontos - centroide
    # Calcula a distância euclidiana de cada ponto até o centroide
    distancias = np.sqrt(pontos_c[0, :]**2 + pontos_c[1, :]**2)
    # Calcula a distância média
    dist_media = np.mean(distancias)

    # Define a escala para que a distância média seja sqrt(2)
    escala = np.sqrt(2) / dist_media

    # Cria a matriz de normalização
    T = np.array([[escala,     0, -escala * centroide[0][0]],
                  [    0, escala, -escala * centroide[1][0]],
                  [    0,     0,                      1]])

    # Adiciona a coordenada homogênea e normaliza os pontos
    pontosh = np.vstack((pontos, np.ones(pontos.shape[1])))
    npontosh = T @ pontosh
    # Remove a última linha (coordenada homogênea)
    npontosh = npontosh[:-1, :]

    return T, npontosh

def meu_DLT(pts1, pts2):
    """
    Implementa o método Direct Linear Transform (DLT) para estimar a homografia.
    Constrói a matriz A a partir das correspondências e resolve via SVD.
    """
    pts1h = np.vstack([pts1, np.ones(pts1.shape[1])])
    pts2h = np.vstack([pts2, np.ones(pts2.shape[1])])

    # Inicializa a matriz A com zeros
    A = np.zeros((2 * pts1.shape[1], 9))
    k = 0
    for i in range(pts1.shape[1]):
        # Primeiro par de equações para cada correspondência
        A[k, 3:6] = -pts2h[2, i] * pts1h[:, i].T
        A[k, 6:9] =  pts2h[1, i] * pts1h[:, i].T
        # Segundo par de equações para cada correspondência
        A[k+1, 0:3] =  pts2h[2, i] * pts1h[:, i].T
        A[k+1, 6:9] = -pts2h[0, i] * pts1h[:, i].T
        k += 2

    # Calcula a decomposição SVD de A e seleciona o vetor associado ao menor valor singular
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    # Reshape do vetor h para a matriz 3x3 de homografia
    H_matrix = h.reshape(3, 3)

    return H_matrix

def minha_homografia(pts1, pts2):
    """
    Calcula a homografia utilizando a normalização dos pontos e o DLT.
    Realiza a desnormalização da matriz final.
    """
    # Normaliza os pontos
    T1, npts1 = normaliza_pontos(pts1)
    T2, npts2 = normaliza_pontos(pts2)

    # Estima a homografia com os pontos normalizados
    Hn = meu_DLT(npts1, npts2)

    # Desnormaliza a homografia
    H = np.linalg.inv(T2) @ Hn @ T1

    return H

def gold_standard(pts1, pts2, N, T_threshold):
    """
    Implementa um método similar ao RANSAC para encontrar a melhor homografia.
    Itera N vezes escolhendo aleatoriamente 4 correspondências e seleciona o modelo com maior número de inliers.
    """
    # Se número de pontos insuficiente, retorna None
    if pts1.shape[1] < 4:
        return None
    n = N           # Número de iterações
    I_total = 0     # Maior quantidade de inliers encontrada
    inliers = None  # Máscara de inliers correspondente

    while n:
        # Seleciona aleatoriamente 4 correspondências
        indices = random.sample(range(pts1.shape[1]), 4)
        pts1_s = pts1[:, indices]
        pts2_s = pts2[:, indices]

        # Calcula a homografia com essas 4 correspondências
        H = minha_homografia(pts1_s, pts2_s)

        # Converte pts1 para coordenadas homogêneas e projeta com H
        pts1h = np.vstack([pts1, np.ones(pts1.shape[1])])
        pts2h_proj = H @ pts1h
        # Evita divisão por zero na última linha
        pts2h_proj[2, :] = np.where(pts2h_proj[2, :] == 0, 1e-10, pts2h_proj[2, :])
        pts2h_proj = pts2h_proj / pts2h_proj[2, :]

        # Calcula a distância euclidiana entre os pontos projetados e os pontos reais (pts2)
        d = np.linalg.norm(pts2h_proj[0:2, :] - pts2, axis=0)

        # Cria uma máscara dos inliers onde a distância é menor que o limiar
        inliers_mask = d < T_threshold
        num_inliers = np.sum(inliers_mask)

        # Atualiza se encontrou modelo com mais inliers
        if num_inliers > I_total:
            I_total = num_inliers
            inliers = inliers_mask

        n -= 1

    # Reestima a homografia usando todas as correspondências inliers
    H_final = minha_homografia(pts1[:, inliers], pts2[:, inliers])

    return H_final

# FUNÇÃO DE DETECÇÃO DE CORRESPONDÊNCIAS

def match_detector(img1, img2):
    """
    Encontra pontos de interesse e correspondências entre duas imagens usando SIFT e FLANN.
    Exibe as correspondências encontradas e retorna as coordenadas dos pontos.
    """
    # Cria o detector SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Configura parâmetros para o FLANN Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Encontra as correspondências com k=2
    matches = flann.knnMatch(des1, des2, k=2)

    # Filtra as correspondências com o ratio test de Lowe
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Desenha as correspondências para visualização
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                  matchColor=(0, 255, 0), flags=2)
    cv2.imshow("Correspondências", matches_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Correspondências")

    # Extrai as coordenadas dos pontos correspondentes
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good])

    return src_pts, dst_pts, matches_img

def registration(img1, img2):
    """
    Realiza o registro entre duas imagens: detecta correspondências e estima a homografia usando o método gold_standard.
    """
    src_pts, dst_pts, _ = match_detector(img1, img2)
    # Transpõe os pontos para o formato (2 x N)
    H = gold_standard(src_pts.T, dst_pts.T, 200, 20)

    return H

# FUNÇÃO PARA CRIAÇÃO DE MÁSCARA

def create_mask(img1, img2, version, smoothing_window_size):
    """
    Cria uma máscara com transição suave para mesclar as duas imagens.
    version: 'left_image' ou 'right_image' para definir a transição.
    """
    # Define as dimensões das imagens de entrada
    height_img1 = img1.shape[0]
    height_img2 = img2.shape[0]
    width_img1  = img1.shape[1]
    width_img2  = img2.shape[1]

    # Define as dimensões do panorama: altura máxima e soma das larguras
    height_panorama = max(height_img1, height_img2)
    width_panorama  = width_img1 + width_img2

    # Offset para a janela de suavização
    offset = int(smoothing_window_size / 2)
    # Barreira definida a partir da largura da primeira imagem
    barrier = width_img1 - offset

    # Cria uma máscara de zeros com tamanho do panorama
    mask = np.zeros((height_panorama, width_panorama), dtype=np.float32)

    if version == 'left_image':
        # Cria uma transição suave na região de sobreposição
        if barrier - offset >= 0:
            x_grad = np.linspace(1, 0, 2 * offset)
            mask[:, barrier - offset:barrier + offset] = np.tile(x_grad, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, :width_img1] = 1
    else:
        # Cria a máscara para a imagem da direita
        if barrier + offset <= width_panorama:
            x_grad = np.linspace(0, 1, 2 * offset)
            mask[:, barrier - offset:barrier + offset] = np.tile(x_grad, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        else:
            mask[:, width_img1:] = 1

    # Converte a máscara para 3 canais para multiplicação com imagens coloridas
    return cv2.merge([mask, mask, mask])

# FUNÇÃO PARA BLENDING DE DUAS IMAGENS

def blending(img1, img2):
    """
    Realiza o blend entre duas imagens.
    Registra as imagens, aplica a homografia à segunda imagem, cria máscaras de suavização e
    soma as imagens mascaradas para formar o panorama.
    """
    # Estima a homografia que transforma img2 no espaço de img1
    H = registration(img1, img2)

    # Obtém as dimensões das imagens
    height_img1, width_img1 = img1.shape[:2]
    height_img2, width_img2 = img2.shape[:2]
    # Define as dimensões do panorama
    height_panorama = max(height_img1, height_img2)
    width_panorama  = width_img1 + width_img2

    # Cria um panorama base com fundo preto (tipo float para operações)
    panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)

    # Cria a máscara para a imagem da esquerda e posiciona img1
    mask1 = create_mask(img1, img2, version='left_image', smoothing_window_size=smoothing_window_size)
    panorama1[0:height_img1, 0:width_img1, :] = img1.astype(np.float32)
    panorama1 *= mask1

    # Cria a máscara para a imagem da direita e aplica a homografia em img2
    mask2 = create_mask(img1, img2, version='right_image', smoothing_window_size=smoothing_window_size)
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)).astype(np.float32)
    panorama2 *= mask2

    # Soma os dois panoramas parciais para realizar o blend
    result = panorama1 + panorama2

    # Garante que os valores dos pixels estejam no intervalo [0, 255] e converte para uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Recorta a região válida do panorama (não preta) usando bounding box
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        final_result = result[y:y+h, x:x+w]
    else:
        final_result = result

    return final_result

# FUNÇÃO PRINCIPAL PARA COSTURA DE UMA LISTA DE IMAGENS

def cria_panorama(lista_imgs):
    """
    Recebe uma lista com os caminhos das imagens e realiza a costura sequencial.
    A cada iteração, a imagem corrente é combinada com o panorama formado até então.
    """
    # Carrega a primeira imagem para iniciar o panorama
    panorama_atual = cv2.imread(lista_imgs[0])

    # Converte para float32 para operações
    panorama_atual = panorama_atual.astype(np.float32)

    # Loop para processar as imagens restantes
    for imagem in lista_imgs[1:]:
        # Carrega a próxima imagem
        nova_img = cv2.imread(imagem)
        nova_img = nova_img.astype(np.float32)

        # Realiza o blend entre o panorama atual e a nova imagem
        panorama_atual = blending(panorama_atual.astype(np.uint8), nova_img.astype(np.uint8))
        cv2.imshow("Panorama Atual", panorama_atual)
        cv2.waitKey(0)
        cv2.destroyWindow("Panorama Atual")

    return panorama_atual

# Define o tamanho da janela de suavização para o blend
smoothing_window_size = 1000

# Lista de caminhos das imagens para formar o panorama
imgs_to_panorama = ['/pasta/01.jpg', '/pasta/02.jpg', '/pasta/03.jpg']

# Cria o panorama com base na lista de imagens
panorama_final = cria_panorama(imgs_to_panorama)

# Salvar o resultado
cv2.imwrite('/pasta/panorama_final.jpg', panorama_final)