import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import io
import os
import socket
import struct
import tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb
import scipy.ndimage as ndimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import inspect
import threading
import traceback


# ==============================================================================
# CONFIGURAÇÃO DO ARQUIVO DE DADOS
# ==============================================================================
SEPARADOR_CSV = ';'
caminho_csv = "C:\\Users\\mateu\\Documents\\Projetos\\ProjetodeOficinadeIntegracao\\melhores_mapeamento\\melhores_mapeamento\\dados113.csv"
# Mapeamento das colunas curtas do CSV novo (conforme notebook atualizado)
# Mantém os nomes originais como chaves para facilitar consulta.
COLUNAS_NOVO_FORMATO = {
    'N': 'N', 't': 't', 'x': 'x', 'y': 'y', 'theta': 'θ',
    'v': 'v', 'omega': 'ω', 'distancia': 'd',
    'erro_frontal': 'ef', 'delta_erro_frontal': 'Δef', 'uc_frontal': 'of',
    'erro_imu': 'ei', 'delta_erro_imu': 'Δei', 'uc_imu': 'oi',
    'duty_e': 'dl', 'duty_d': 'dr',
    'encoder_e': 'el', 'encoder_d': 'er',
    'posicao_frontal': 'fp',
    'linha_esq': 'ld', 'linha_dir': 'rd',
}
SENSORES_FRONTAIS = [f's{i}' for i in range(1, 14)]


# ==============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================
def carregar_dados(caminho, sep=SEPARADOR_CSV):
    dados_original = pd.read_csv(caminho, sep=sep)
    dados = pd.read_csv(caminho, sep=sep)
    dados['y'] = dados['y'] - dados['y'].iloc[0]
    dados['θ'] = dados['θ'] - dados['θ'].iloc[0]
    return dados_original, dados


def carregar_csv_de_teste(caminho, sep=SEPARADOR_CSV):
    """
    Carrega um CSV usando a variável caminho_csv para testar os gráficos.
    Se o caminho for uma pasta, escolhe o primeiro arquivo .csv disponível.
    """
    if os.path.isdir(caminho):
        arquivos_csv = sorted(
            f for f in os.listdir(caminho)
            if os.path.isfile(os.path.join(caminho, f)) and f.lower().endswith('.csv')
        )
        if not arquivos_csv:
            raise FileNotFoundError(f"Nenhum CSV encontrado em: {caminho}")
        caminho = os.path.join(caminho, arquivos_csv[0])

    dados_original, dados = carregar_dados(caminho, sep=sep)
    calc = preparar_dados(dados, dados_original)
    return dados, dados_original, calc, caminho


# ==============================================================================
# FUNÇÕES AUXILIARES (filtros, curvaturas, etc.) — versão do notebook atualizado
# ==============================================================================

def gaussian_filter1d_melhorado(data, sigma, endpoint_preservation_ratio=0.1):
    """Filtro Gaussiano com preservação suave dos endpoints (transição linear)."""
    if hasattr(data, 'values'):
        data_array = data.values.astype(float)
    else:
        data_array = np.array(data, dtype=float)

    if sigma <= 0:
        return data_array

    window_size = int(6 * sigma + 1)
    if window_size % 2 == 0:
        window_size += 1
    x = np.arange(window_size) - window_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    filtered_data = np.convolve(data_array, kernel, mode='same').astype(float)

    n_preserve = int(len(data_array) * endpoint_preservation_ratio)
    if n_preserve > 0:
        n_preserve = min(n_preserve, len(data_array) // 4)
        for i in range(n_preserve):
            weight = (n_preserve - i) / n_preserve
            filtered_data[i] = weight * data_array[i] + (1 - weight) * filtered_data[i]
        for i in range(n_preserve):
            idx = len(data_array) - 1 - i
            weight = (n_preserve - i) / n_preserve
            filtered_data[idx] = weight * data_array[idx] + (1 - weight) * filtered_data[idx]
    else:
        filtered_data[0] = data_array[0]
        filtered_data[-1] = data_array[-1]
    return filtered_data


def gaussian_filter_1d_custom(data, sigma, preserve_endpoints=True):
    """Filtro Gaussiano (scipy) com preservação de endpoints e transição suave."""
    filtered_data = ndimage.gaussian_filter1d(data, sigma=sigma)
    if preserve_endpoints:
        filtered_data[0] = data[0]
        filtered_data[-1] = data[-1]
        n_transition = max(1, int(len(data) * 0.1))
        for i in range(min(n_transition, len(data) // 4)):
            weight = i / n_transition
            filtered_data[i] = (1 - weight) * data[i] + weight * filtered_data[i]
        for i in range(min(n_transition, len(data) // 4)):
            idx = len(data) - 1 - i
            weight = i / n_transition
            filtered_data[idx] = (1 - weight) * data[idx] + weight * filtered_data[idx]
    return filtered_data


def aplicar_filtro_trajetoria_completa(x_data, y_data, sigma, endpoint_preservation_ratio=0.1):
    """Aplica filtro Gaussiano na trajetória completa (X e Y)."""
    x_filtered = gaussian_filter1d_melhorado(x_data, sigma, endpoint_preservation_ratio)
    y_filtered = gaussian_filter1d_melhorado(y_data, sigma, endpoint_preservation_ratio)
    return x_filtered, y_filtered


def aplicar_filtro_gaussiano_seletivo(x_data, y_data, lista_configs):
    """Aplica filtro Gaussiano apenas em trechos selecionados da trajetória."""
    x_filtered = x_data.copy() if hasattr(x_data, 'copy') else np.array(x_data).copy()
    y_filtered = y_data.copy() if hasattr(y_data, 'copy') else np.array(y_data).copy()
    use_pandas = hasattr(x_data, 'iloc')
    n_total = len(x_data)

    for config in lista_configs:
        inicio = config['inicio']
        fim = config['fim']
        sigma = config.get('sigma', 2.0)
        preserve_endpoints = config.get('preserve_endpoints', True)

        if inicio >= n_total or fim >= n_total or inicio >= fim:
            print(f"[AVISO] Trecho ignorado: índices [{inicio}:{fim}] inválidos para dados com {n_total} linhas.")
            continue

        if use_pandas:
            x_trecho = x_data.iloc[inicio:fim + 1].values
            y_trecho = y_data.iloc[inicio:fim + 1].values
        else:
            x_trecho = x_data[inicio:fim + 1]
            y_trecho = y_data[inicio:fim + 1]

        if len(x_trecho) == 0:
            continue

        x_suave = gaussian_filter_1d_custom(x_trecho, sigma, preserve_endpoints)
        y_suave = gaussian_filter_1d_custom(y_trecho, sigma, preserve_endpoints)

        if use_pandas:
            x_filtered.iloc[inicio:fim + 1] = x_suave
            y_filtered.iloc[inicio:fim + 1] = y_suave
        else:
            x_filtered[inicio:fim + 1] = x_suave
            y_filtered[inicio:fim + 1] = y_suave

    return np.array(x_filtered), np.array(y_filtered)


def calcular_curvatura(x, y):
    """Calcula a curvatura ao longo da trajetória."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** (3 / 2)
    denominator[denominator < 1e-9] = 1e-9
    return numerator / denominator


def calcular_pwm_por_curvatura(x_suave, y_suave, pwm_min, pwm_max, k, p,
                                curv_percentil_min, curv_percentil_max,
                                threshold_curva, pontos_frenagem, pwm_frenagem):
    """Calcula o perfil de PWM baseado na curvatura da trajetória (notebook atualizado)."""
    curvatura = calcular_curvatura(x_suave, y_suave)
    curv_min = np.percentile(curvatura, curv_percentil_min)
    curv_max = np.percentile(curvatura, curv_percentil_max)
    if curv_max <= curv_min:
        curv_max = curv_min + 1e-9
    curvatura_norm = np.clip((curvatura - curv_min) / (curv_max - curv_min), 0, 1)
    pwm_base = pwm_min + (pwm_max - pwm_min) * np.exp(-k * (curvatura_norm ** p))

    em_curva = curvatura_norm > threshold_curva
    inicio_curvas = np.where(np.diff(em_curva.astype(int)) == 1)[0] + 1
    fim_curvas = np.where(np.diff(em_curva.astype(int)) == -1)[0] + 1

    pwm_ajustado = pwm_base.copy()
    for inicio in inicio_curvas:
        for i in range(max(0, inicio - pontos_frenagem), inicio):
            if i < len(pwm_ajustado):
                distancia_curva = inicio - i
                fator_frenagem = distancia_curva / pontos_frenagem
                if distancia_curva <= pontos_frenagem:
                    pwm_ajustado[i] = pwm_frenagem + (pwm_base[i] - pwm_frenagem) * (1 - fator_frenagem)

    pwm_finais = np.clip(pwm_ajustado, pwm_min, pwm_max)
    return {
        'pwm': pwm_finais,
        'curvatura': curvatura,
        'curvatura_normalizada': curvatura_norm,
        'em_curva': em_curva,
        'inicio_curvas': inicio_curvas,
        'fim_curvas': fim_curvas,
        'pwm_originais': pwm_base,
        'pwm_ajustados': pwm_ajustado,
        'parametros_utilizados': {
            'pwm_min': pwm_min, 'pwm_max': pwm_max,
            'k': k, 'p': p, 'threshold_curva': threshold_curva,
            'pontos_frenagem': pontos_frenagem, 'pwm_frenagem': pwm_frenagem
        }
    }


def reamostrar_por_distancia_com_extensao(x_suave, y_suave, pwm_pontos, distancia_alvo,
                                           extra_end_points=0, pontos_extensao=0,
                                           dist_extensao=100.0):
    """
    Reamostragem por distância (versão do notebook atualizado), com opção de:
      - incluir N pontos adicionais antes do fim (extra_end_points)
      - estender a trajetória após o último ponto (pontos_extensao / dist_extensao)
    """
    indices_selecionados = [0]
    ultimo_x = x_suave[0]
    ultimo_y = y_suave[0]

    for i in range(1, len(x_suave)):
        dist = np.sqrt((x_suave[i] - ultimo_x) ** 2 + (y_suave[i] - ultimo_y) ** 2)
        if dist >= distancia_alvo:
            indices_selecionados.append(i)
            ultimo_x = x_suave[i]
            ultimo_y = y_suave[i]

    if indices_selecionados[-1] != len(x_suave) - 1:
        indices_selecionados.append(len(x_suave) - 1)

    extras = []
    for k in range(1, extra_end_points + 1):
        idx = len(x_suave) - 1 - k
        if idx >= 0 and idx not in indices_selecionados:
            extras.append(idx)
    indices_selecionados += extras
    indices_selecionados = sorted(indices_selecionados)

    x_reamostrado = [x_suave[i] for i in indices_selecionados]
    y_reamostrado = [y_suave[i] for i in indices_selecionados]
    pwm_reamostrado = [pwm_pontos[i] for i in indices_selecionados]
    indices_dist_ext = list(indices_selecionados)
    dists_ext = []

    if pontos_extensao > 0:
        if len(x_reamostrado) >= 2:
            dx = x_reamostrado[-1] - x_reamostrado[-2]
            dy = y_reamostrado[-1] - y_reamostrado[-2]
            dist_ultimo_seg = np.sqrt(dx ** 2 + dy ** 2)
            if dist_ultimo_seg > 0:
                dx_norm = dx / dist_ultimo_seg
                dy_norm = dy / dist_ultimo_seg
            else:
                dx_norm, dy_norm = 1.0, 0.0
        else:
            dx_norm, dy_norm = 1.0, 0.0

        ultimo_x_ext = x_reamostrado[-1]
        ultimo_y_ext = y_reamostrado[-1]
        ultimo_pwm_ext = pwm_reamostrado[-1]

        for i in range(pontos_extensao):
            novo_x = ultimo_x_ext + dx_norm * dist_extensao
            novo_y = ultimo_y_ext + dy_norm * dist_extensao
            novo_pwm = ultimo_pwm_ext

            x_reamostrado.append(novo_x)
            y_reamostrado.append(novo_y)
            pwm_reamostrado.append(novo_pwm)
            dists_ext.append(dist_extensao * (i + 1))

            ultimo_x_ext = novo_x
            ultimo_y_ext = novo_y
            ultimo_pwm_ext = novo_pwm

    return (np.array(x_reamostrado), np.array(y_reamostrado), np.array(pwm_reamostrado),
            indices_dist_ext, dists_ext)


# ==============================================================================
# ANÁLISE DE MARCAS (linha esquerda/direita) — versão do notebook atualizado
# ==============================================================================
def contar_marcas_transicoes(left_detected, right_detected):
    """Detecta marcas (transições 0→1 e 1→0) considerando ld OU rd ativos."""
    left_array = left_detected.values if hasattr(left_detected, 'values') else np.array(left_detected)
    right_array = right_detected.values if hasattr(right_detected, 'values') else np.array(right_detected)

    marca_detectada = (left_array == 1) | (right_array == 1)
    diff_marca = np.diff(marca_detectada.astype(int))

    inicios_marca = np.where(diff_marca == 1)[0] + 1
    fins_marca = np.where(diff_marca == -1)[0]

    if marca_detectada[0]:
        inicios_marca = np.insert(inicios_marca, 0, 0)
    if marca_detectada[-1]:
        fins_marca = np.append(fins_marca, len(marca_detectada) - 1)

    return marca_detectada, inicios_marca, fins_marca


# ==============================================================================
# PREPARAÇÃO DOS DADOS PARA OS GRÁFICOS — adaptado ao formato novo do CSV
# ==============================================================================
def preparar_dados(dados, dados_original):
    """Calcula todas as variáveis derivadas usadas nos gráficos (colunas novas)."""
    distancia = dados['d'] if 'd' in dados.columns else pd.Series(dados.index)

    # --- Marcas (linha esquerda/direita detectada) ---
    marca_detectada, inicios_marca, fins_marca = contar_marcas_transicoes(
        dados['ld'], dados['rd']
    )
    marcas_info = []
    for num, (ini, fim) in enumerate(zip(inicios_marca, fins_marca), start=1):
        x_medio = dados['x'].iloc[ini:fim + 1].mean()
        y_medio = dados['y'].iloc[ini:fim + 1].mean()
        marcas_info.append({'marca_num': num, 'x_medio': x_medio, 'y_medio': y_medio})
    df_marcas = pd.DataFrame(marcas_info)

    # --- Encoders ---
    encoder_e = dados['el']
    encoder_d = dados['er']
    pulsos_e_diff = encoder_e.diff().fillna(0)
    pulsos_d_diff = encoder_d.diff().fillna(0)
    encoder_diferencial = encoder_e - encoder_d
    delta_diff_enc = pulsos_e_diff - pulsos_d_diff
    diff_absoluta = np.abs(delta_diff_enc)
    outliers_pulsos = diff_absoluta > (np.mean(diff_absoluta) + 3 * np.std(diff_absoluta))

    # --- Controle frontal (substitui o antigo "óptico") ---
    erro_frontal = dados['ef']
    delta_erro_frontal = dados['Δef']
    uc_frontal = dados['of']
    posicao_frontal = dados['fp']
    POSICAO_MAX = 2100
    POSICAO_MIN = -2100

    # --- Duty cycle dos motores ---
    duty_e = dados['dl']
    duty_d = dados['dr']

    # --- Sensores frontais s1..s13 ---
    sensores_disponiveis = [s for s in SENSORES_FRONTAIS if s in dados.columns]

    # --- Controle IMU ---
    erro_imu = dados['ei']
    delta_erro_imu = dados['Δei']
    uc_imu = dados['oi']

    # --- Velocidades e orientação ---
    theta_rad = dados['θ']
    v_linear = dados['v']
    v_angular = dados['ω']

    # --- Suavização da trajetória (parâmetros do notebook atualizado) ---
    sigma_1 = 10
    trechos_gaussiano = [
        {'inicio': 300, 'fim': 450, 'sigma': 10.0, 'preserve_endpoints': True},
    ]
    sigma_2 = 10

    x_gaus, y_gaus = aplicar_filtro_trajetoria_completa(dados['x'], dados['y'], sigma_1, 0.1)
    x_suave, y_suave = aplicar_filtro_gaussiano_seletivo(x_gaus, y_gaus, trechos_gaussiano)
    x_suave, y_suave = aplicar_filtro_trajetoria_completa(x_suave, y_suave, sigma_2, 0.1)

    # --- PWM por curvatura (parâmetros do notebook atualizado) ---
    resultado_pwm = calcular_pwm_por_curvatura(
        x_suave, y_suave, pwm_min=3000, pwm_max=4000,
        k=6.0, p=2.0, curv_percentil_min=10, curv_percentil_max=90,
        threshold_curva=0.8, pontos_frenagem=10, pwm_frenagem=0
    )
    pwm_pontos = resultado_pwm['pwm']

    # --- Reamostragem final por distância (parâmetros do notebook atualizado) ---
    distancia_alvo = 20.0
    x_dist, y_dist, pwm_dist, indices_dist_ext, dists_ext = reamostrar_por_distancia_com_extensao(
        x_suave, y_suave, pwm_pontos, distancia_alvo,
        extra_end_points=0, pontos_extensao=0, dist_extensao=50.0
    )

    return {
        'distancia': distancia,
        'marca_detectada': marca_detectada,
        'df_marcas': df_marcas,
        'encoder_e': encoder_e, 'encoder_d': encoder_d,
        'pulsos_e_diff': pulsos_e_diff, 'pulsos_d_diff': pulsos_d_diff,
        'encoder_diferencial': encoder_diferencial,
        'delta_diff_enc': delta_diff_enc,
        'outliers_pulsos': outliers_pulsos,
        'corr_encoders': np.corrcoef(encoder_e, encoder_d)[0, 1],
        'corr_pulsos': np.corrcoef(pulsos_e_diff, pulsos_d_diff)[0, 1],
        'erro_frontal': erro_frontal, 'delta_erro_frontal': delta_erro_frontal,
        'uc_frontal': uc_frontal, 'posicao_frontal': posicao_frontal,
        'POSICAO_MAX': POSICAO_MAX, 'POSICAO_MIN': POSICAO_MIN,
        'duty_e': duty_e, 'duty_d': duty_d,
        'sensores_disponiveis': sensores_disponiveis,
        'erro_imu': erro_imu, 'delta_erro_imu': delta_erro_imu, 'uc_imu': uc_imu,
        'theta_rad': theta_rad, 'v_linear': v_linear, 'v_angular': v_angular,
        'x_suave': x_suave, 'y_suave': y_suave,
        'resultado_pwm': resultado_pwm, 'pwm_pontos': pwm_pontos,
        'x_dist': x_dist, 'y_dist': y_dist, 'pwm_dist': pwm_dist,
        'distancia_alvo': distancia_alvo,
        'trechos_gaussiano': trechos_gaussiano,
    }


# ==============================================================================
# FUNÇÕES DE PLOTAGEM — cada função plota em uma figura já criada (fig, axes)
# ==============================================================================

# ── GRÁFICO 1: Trajetória com marcas detectadas (linha esq/dir) ───────────────
def plot_trajetoria_marcas(fig, dados, calc):
    ax = fig.add_subplot(111)
    ax.plot(dados['x'], dados['y'], 'b-', alpha=0.5, linewidth=2, label='Trajetória')

    mask = calc['marca_detectada']
    if mask.any():
        ax.scatter(dados['x'][mask], dados['y'][mask],
                   c='red', s=30, alpha=0.6, label='Marca Detectada (ld/rd)', zorder=5)

    for _, marca in calc['df_marcas'].iterrows():
        ax.scatter(marca['x_medio'], marca['y_medio'],
                   c='yellow', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)
        ax.text(marca['x_medio'], marca['y_medio'],
                f"M{int(marca['marca_num'])}", fontsize=12, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1))

    ax.scatter(dados['x'].iloc[0], dados['y'].iloc[0], color='green', s=200, marker='o',
               label='Início', zorder=10, edgecolors='white', linewidth=3)
    ax.scatter(dados['x'].iloc[-1], dados['y'].iloc[-1], color='purple', s=200, marker='X',
               label='Fim', zorder=10, edgecolors='white', linewidth=3)
    ax.set_xlabel('Posição X (mm)', fontsize=12)
    ax.set_ylabel('Posição Y (mm)', fontsize=12)
    ax.set_title('Trajetória com Marcas Detectadas (Linha Esq/Dir)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')
    fig.tight_layout()


# ── GRÁFICO 2: Trajetória planejada vs realizada ───────────────────────────────
def plot_trajetoria_planejada_realizada(fig, dados, dados_original, calc):
    ax = fig.add_subplot(111)
    ax.plot(dados_original['x'], dados_original['y'], 'r-', linewidth=1, alpha=1,
            label='Trajetória Planejada')
    ax.scatter(dados['x'], dados['y'], c='blue', s=8, alpha=0.6, label='Trajetória Realizada')
    ax.scatter(dados['x'].iloc[0], dados['y'].iloc[0], color='green', s=150, marker='o',
               label='Início', zorder=10, edgecolors='white', linewidth=2)
    ax.scatter(dados['x'].iloc[-1], dados['y'].iloc[-1], color='red', s=150, marker='X',
               label='Fim', zorder=10, edgecolors='white', linewidth=2)
    ax.set_title('Trajetória da Pista - Planejada vs Realizada', fontsize=14, fontweight='bold')
    ax.set_xlabel('Posição X (mm)', fontsize=12)
    ax.set_ylabel('Posição Y (mm)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.minorticks_on()
    ax.set_aspect('equal')
    fig.tight_layout()


# ── GRÁFICO 3: Encoders acumulados (2x2) ──────────────────────────────────────
def plot_encoders_acumulados(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('ANÁLISE DOS ENCODERS ACUMULADOS', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    encoder_e = calc['encoder_e']; encoder_d = calc['encoder_d']
    encoder_diff = encoder_e - encoder_d

    axes[0, 0].plot(x_axis, encoder_e, 'b-', linewidth=1.5, alpha=0.8, label='Encoder E')
    axes[0, 0].plot(x_axis, encoder_d, 'r-', linewidth=1.5, alpha=0.8, label='Encoder D')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Pulsos Acumulados')
    axes[0, 0].set_title('Evolução dos Encoders'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, encoder_diff, 'purple', linewidth=1.5)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].axhline(encoder_diff.mean(), color='red', linestyle='--',
                       label=f'Média: {encoder_diff.mean():.1f}')
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Diferença (E - D)')
    axes[0, 1].set_title('Diferença Entre Encoders'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(encoder_e, encoder_d, alpha=0.6, s=8, c=x_axis, cmap='viridis')
    z = np.polyfit(encoder_e, encoder_d, 1)
    axes[1, 0].plot(encoder_e, np.poly1d(z)(encoder_e), 'r-', alpha=0.8, linewidth=2,
                    label=f'r={calc["corr_encoders"]:.4f}')
    min_v = min(encoder_e.min(), encoder_d.min()); max_v = max(encoder_e.max(), encoder_d.max())
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, label='Sincronia Perfeita')
    axes[1, 0].set_xlabel('Encoder E'); axes[1, 0].set_ylabel('Encoder D')
    axes[1, 0].set_title('Correlação Entre Encoders'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(encoder_diff, bins=40, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(encoder_diff.mean(), color='red', linestyle='--',
                       label=f'Média: {encoder_diff.mean():.1f}')
    axes[1, 1].axvline(encoder_diff.median(), color='green', linestyle='--',
                       label=f'Mediana: {encoder_diff.median():.1f}')
    axes[1, 1].set_xlabel('Diferença (E - D)'); axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição da Diferença'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 4: Pulsos incrementais dos encoders (2x2) ─────────────────────────
def plot_pulsos_incrementais(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('ANÁLISE DOS PULSOS ENTRE AMOSTRAS', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    pulsos_e_diff = calc['pulsos_e_diff']; pulsos_d_diff = calc['pulsos_d_diff']
    delta_diff = calc['delta_diff_enc']; outliers_mask = calc['outliers_pulsos']

    axes[0, 0].plot(x_axis, pulsos_e_diff, 'b-', linewidth=1, alpha=0.8, label='Δ Encoder E')
    axes[0, 0].plot(x_axis, pulsos_d_diff, 'r-', linewidth=1, alpha=0.8, label='Δ Encoder D')
    axes[0, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Pulsos por Amostra')
    axes[0, 0].set_title('Pulsos Incrementais'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, delta_diff, 'purple', linewidth=1.5)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].axhline(delta_diff.mean(), color='red', linestyle='--',
                       label=f'Média: {delta_diff.mean():.3f}')
    if np.any(outliers_mask):
        axes[0, 1].scatter(x_axis[outliers_mask], delta_diff[outliers_mask],
                           color='red', s=20, alpha=0.8,
                           label=f'Anomalias ({np.sum(outliers_mask)})')
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Δ(E - D)')
    axes[0, 1].set_title('Diferença dos Pulsos'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(pulsos_e_diff, pulsos_d_diff, alpha=0.6, s=8, c=x_axis, cmap='plasma')
    z = np.polyfit(pulsos_e_diff, pulsos_d_diff, 1)
    axes[1, 0].plot(pulsos_e_diff, np.poly1d(z)(pulsos_e_diff), 'r-', alpha=0.8, linewidth=2,
                    label=f'r={calc["corr_pulsos"]:.4f}')
    min_v = min(pulsos_e_diff.min(), pulsos_d_diff.min()); max_v = max(pulsos_e_diff.max(), pulsos_d_diff.max())
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('Δ Encoder E'); axes[1, 0].set_ylabel('Δ Encoder D')
    axes[1, 0].set_title('Correlação dos Pulsos'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    bp = axes[1, 1].boxplot([pulsos_e_diff, pulsos_d_diff, delta_diff],
                            labels=['Δ E', 'Δ D', 'Diferença'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'red', 'purple']):
        patch.set_facecolor(color)
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_ylabel('Pulsos'); axes[1, 1].set_title('Box Plot Pulsos'); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 5: Orientação (θ) — evolução e distribuição ───────────────────────
def plot_orientacao_theta(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('ANÁLISE DA ORIENTAÇÃO (θ)', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    theta_rad = calc['theta_rad']
    theta_graus = theta_rad * 180.0 / pi

    axes[0, 0].plot(x_axis, theta_graus, 'b-', linewidth=1.2, alpha=0.8)
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('θ (graus)')
    axes[0, 0].set_title('Evolução de θ'); axes[0, 0].grid(True, alpha=0.3)

    diff_theta = np.diff(theta_rad) * 180 / pi
    axes[0, 1].plot(x_axis[1:], diff_theta, 'orange', linewidth=1, alpha=0.8)
    axes[0, 1].axhline(0, color='black', alpha=0.5)
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Δθ (graus)')
    axes[0, 1].set_title('Variação de θ por Amostra'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(theta_graus, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].axvline(theta_graus.mean(), color='red', linestyle='--',
                       label=f'Média: {theta_graus.mean():.1f}°')
    axes[1, 0].set_xlabel('θ (graus)'); axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição de θ'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    sc = axes[1, 1].scatter(dados['x'], dados['y'], c=theta_graus, cmap='twilight', s=10, alpha=0.8)
    axes[1, 1].set_xlabel('X (mm)'); axes[1, 1].set_ylabel('Y (mm)')
    axes[1, 1].set_title('Mapa de Orientação'); axes[1, 1].set_aspect('equal')
    fig.colorbar(sc, ax=axes[1, 1], shrink=0.8).set_label('θ (graus)')
    fig.tight_layout()


# ── GRÁFICO 6: Velocidades linear e angular ────────────────────────────────────
def plot_velocidades(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('ANÁLISE DE VELOCIDADES', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    v_lin = calc['v_linear']; v_ang = calc['v_angular'] * 180 / pi

    axes[0, 0].plot(x_axis, v_lin, 'g-', linewidth=1.2, alpha=0.8)
    axes[0, 0].axhline(v_lin.mean(), color='red', linestyle='--', label=f'Média: {v_lin.mean():.3f}')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Velocidade Linear (m/s)')
    axes[0, 0].set_title('Velocidade Linear'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, v_ang, 'm-', linewidth=1.2, alpha=0.8)
    axes[0, 1].axhline(0, color='black', alpha=0.5)
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Velocidade Angular (°/s)')
    axes[0, 1].set_title('Velocidade Angular'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(v_lin, bins=40, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Velocidade Linear (m/s)'); axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição Vel. Linear'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(v_lin, v_ang, alpha=0.5, s=8, c=x_axis, cmap='viridis')
    axes[1, 1].set_xlabel('Vel. Linear (m/s)'); axes[1, 1].set_ylabel('Vel. Angular (°/s)')
    axes[1, 1].set_title('Vel. Linear vs Angular'); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 7: Controle frontal - evolução temporal (2x2) ──────────────────────
def plot_controle_frontal_temporal(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('CONTROLE FRONTAL - EVOLUÇÃO TEMPORAL', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    erro = calc['erro_frontal']; delta_erro = calc['delta_erro_frontal']
    uc = calc['uc_frontal']; posicao = calc['posicao_frontal']

    axes[0, 0].plot(x_axis, erro, 'r-', linewidth=1.2, alpha=0.8)
    axes[0, 0].axhline(0, color='black', alpha=0.5)
    axes[0, 0].axhline(erro.mean(), color='blue', linestyle='--', label=f'Média: {erro.mean():.2f}')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Erro Frontal')
    axes[0, 0].set_title('Evolução do Erro'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, uc, 'g-', linewidth=1.2, alpha=0.8)
    axes[0, 1].axhline(0, color='black', alpha=0.5)
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('UC Frontal')
    axes[0, 1].set_title('Sinal de Controle'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_axis, posicao, 'purple', linewidth=1.2, alpha=0.8)
    axes[1, 0].axhline(calc['POSICAO_MAX'], color='red', linestyle=':', linewidth=2)
    axes[1, 0].axhline(calc['POSICAO_MIN'], color='red', linestyle=':', linewidth=2)
    axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel('Posição')
    axes[1, 0].set_title('Posição Detectada'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(erro, bins=40, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].axvline(0, color='black', alpha=0.5)
    axes[1, 1].set_xlabel('Erro Frontal'); axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição do Erro'); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 8: Controle frontal - correlações ──────────────────────────────────
def plot_controle_frontal_correlacoes(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('CONTROLE FRONTAL - CORRELAÇÕES', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']
    erro = calc['erro_frontal']; uc = calc['uc_frontal']; posicao = calc['posicao_frontal']

    corr_pu = np.corrcoef(posicao, uc)[0, 1]
    corr_pe = np.corrcoef(posicao, erro)[0, 1]

    axes[0, 0].scatter(posicao, uc, alpha=0.6, s=8, c=x_axis, cmap='plasma')
    z = np.polyfit(posicao, uc, 1)
    axes[0, 0].plot(posicao, np.poly1d(z)(posicao), 'r-', linewidth=2, label=f'r={corr_pu:.3f}')
    axes[0, 0].axvline(calc['POSICAO_MAX'], color='red', linestyle=':', alpha=0.8)
    axes[0, 0].axvline(calc['POSICAO_MIN'], color='red', linestyle=':', alpha=0.8)
    axes[0, 0].set_xlabel('Posição'); axes[0, 0].set_ylabel('UC')
    axes[0, 0].set_title('Posição vs UC'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(posicao, erro, alpha=0.6, s=8, c=x_axis, cmap='coolwarm')
    z = np.polyfit(posicao, erro, 1)
    axes[0, 1].plot(posicao, np.poly1d(z)(posicao), 'r-', linewidth=2, label=f'r={corr_pe:.3f}')
    axes[0, 1].set_xlabel('Posição'); axes[0, 1].set_ylabel('Erro')
    axes[0, 1].set_title('Posição vs Erro'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(erro, bins=40, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].axvline(erro.mean(), color='blue', linestyle='--', label=f'Média: {erro.mean():.2f}')
    axes[1, 0].set_xlabel('Erro'); axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição do Erro'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    total_p = len(posicao)
    zonas_vals = [
        np.sum(np.abs(posicao) <= 150),
        np.sum((np.abs(posicao) > 150) & (np.abs(posicao) <= 600)),
        np.sum((np.abs(posicao) > 600) & (np.abs(posicao) <= 1200)),
        np.sum((np.abs(posicao) > 1200) & (np.abs(posicao) <= 2100)),
        np.sum(np.abs(posicao) > 2100),
    ]
    labels_z = ['Centro\n±150', 'Leve\n150-600', 'Int\n600-1200', 'Risco\n1200-2100', 'Crit\n>2100']
    colors_z = ['green', 'yellow', 'orange', 'red', 'black']
    bars = axes[1, 1].bar(labels_z, zonas_vals, color=colors_z, alpha=0.7)
    for bar, v in zip(bars, zonas_vals):
        h = bar.get_height()
        if h > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., h,
                            f'{v}\n({v / total_p * 100:.1f}%)', ha='center', va='bottom', fontsize=8)
    axes[1, 1].set_ylabel('Pontos'); axes[1, 1].set_title('Zonas de Posição')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    fig.tight_layout()


# ── GRÁFICO 9: Duty cycle dos motores (2x2) ───────────────────────────────────
def plot_duty_cycle(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('DUTY CYCLE DOS MOTORES', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    duty_e = calc['duty_e']; duty_d = calc['duty_d']
    duty_medio = (duty_e + duty_d) / 2
    duty_dif = duty_e - duty_d

    axes[0, 0].plot(x_axis, duty_e, 'b-', linewidth=1.2, alpha=0.8, label='Motor E')
    axes[0, 0].plot(x_axis, duty_d, 'r-', linewidth=1.2, alpha=0.8, label='Motor D')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Duty Cycle')
    axes[0, 0].set_title('Duty Cycle dos Motores'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, duty_medio, 'g-', linewidth=1.2, alpha=0.8)
    axes[0, 1].axhline(duty_medio.mean(), color='blue', linestyle='--',
                       label=f'Média: {duty_medio.mean():.1f}')
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Duty Médio')
    axes[0, 1].set_title('Duty Cycle Médio (Velocidade)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_axis, duty_dif, 'purple', linewidth=1.2, alpha=0.8)
    axes[1, 0].axhline(0, color='black', alpha=0.5)
    axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel('Duty Diferencial')
    axes[1, 0].set_title('Duty Diferencial (Direção)'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(duty_e, bins=40, alpha=0.5, color='blue', label='Motor E')
    axes[1, 1].hist(duty_d, bins=40, alpha=0.5, color='red', label='Motor D')
    axes[1, 1].set_xlabel('Duty Cycle'); axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição Duty Cycle'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 10: Heatmap sensores frontais ─────────────────────────────────────
def plot_sensores_heatmap(fig, dados, calc):
    ax = fig.add_subplot(111)
    sensores = calc['sensores_disponiveis']
    if not sensores:
        ax.text(0.5, 0.5, 'Sensores frontais não encontrados', ha='center', va='center')
        return
    x_axis = calc['distancia']
    dados_matriz = dados[sensores].T.values.astype(float)
    dados_norm = np.zeros_like(dados_matriz)
    for i in range(dados_matriz.shape[0]):
        s_min = dados_matriz[i].min(); s_max = dados_matriz[i].max()
        dados_norm[i] = (dados_matriz[i] - s_min) / (s_max - s_min) if s_max > s_min else 0.5
    im = ax.imshow(dados_norm, aspect='auto', cmap='inferno',
                   extent=[x_axis.min(), x_axis.max(), 0, len(sensores)], vmin=0, vmax=1)
    ax.set_xlabel('Distância (mm)', fontsize=12); ax.set_ylabel('Sensores', fontsize=12)
    ax.set_title('HEATMAP DOS SENSORES FRONTAIS (s1 a s13)', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(sensores)))
    ax.set_yticklabels(sensores, fontsize=10)
    ax.invert_yaxis()
    ax.axhline(6.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='s7 (Centro)')
    ax.legend(loc='upper right', fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Valor Normalizado (0=Linha, 1=Piso)', fontsize=11)
    fig.tight_layout()


# ── GRÁFICO 11: Análise posição da linha (sensores frontais) (2x2) ─────────────
def plot_sensores_linha(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('ANÁLISE DA POSIÇÃO DA LINHA (SENSORES FRONTAIS)', fontsize=14, fontweight='bold')
    sensores = calc['sensores_disponiveis']
    if not sensores:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
        return
    x_axis = calc['distancia']; x_label = 'Distância (mm)'

    limiares = {s: dados[s].mean() for s in sensores}
    deteccoes_preto = {s: dados[s] < limiares[s] for s in sensores}
    n_sens = len(sensores)
    centro = n_sens // 2

    def calc_pos_linha():
        posicoes, intensidades = [], []
        for _, row in dados[sensores].iterrows():
            num = 0; den = 0
            for j, s in enumerate(sensores):
                pos_s = j - centro
                val_inv = max(0, limiares[s] - row[s])
                num += pos_s * val_inv; den += val_inv
            posicoes.append(num / den if den > 0 else 0)
            intensidades.append(den)
        return np.array(posicoes), np.array(intensidades)

    posicoes_linha, intensidades_linha = calc_pos_linha()
    sensores_ativos = np.sum([deteccoes_preto[s] for s in sensores], axis=0)

    axes[0, 0].plot(x_axis, posicoes_linha, 'b-', linewidth=1.2, alpha=0.8)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7, label='Centro')
    axes[0, 0].axhline(np.mean(posicoes_linha), color='green', linestyle='--',
                       label=f'Média: {np.mean(posicoes_linha):.2f}')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Posição da Linha')
    axes[0, 0].set_title('Posição da Linha'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, sensores_ativos, 'purple', linewidth=1.2)
    axes[0, 1].axhline(3, color='green', linestyle='--', alpha=0.7, label='Ideal (3)')
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Sensores Ativos')
    axes[0, 1].set_title('Sensores Detectando Linha'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_axis, intensidades_linha, 'orange', linewidth=1.2)
    axes[1, 0].axhline(np.mean(intensidades_linha), color='red', linestyle='--',
                       label=f'Média: {np.mean(intensidades_linha):.1f}')
    axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel('Intensidade')
    axes[1, 0].set_title('Força do Sinal da Linha'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(posicoes_linha, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(np.mean(posicoes_linha), color='red', linestyle='--',
                       label=f'Média: {np.mean(posicoes_linha):.2f}')
    axes[1, 1].axvline(0, color='green', linestyle='--', label='Centro')
    axes[1, 1].set_xlabel('Posição da Linha'); axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição da Posição'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 12: Controle IMU - evolução temporal (2x2) ────────────────────────
def plot_controle_imu_temporal(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('CONTROLE IMU - EVOLUÇÃO TEMPORAL', fontsize=14, fontweight='bold')
    x_axis = calc['distancia']; x_label = 'Distância (mm)'
    erro = calc['erro_imu']; delta_erro = calc['delta_erro_imu']; uc = calc['uc_imu']
    curvatura = calcular_curvatura(dados['x'].values, dados['y'].values)

    axes[0, 0].plot(x_axis, erro, 'r-', linewidth=1.2, alpha=0.8)
    axes[0, 0].axhline(0, color='black', alpha=0.5)
    axes[0, 0].axhline(erro.mean(), color='blue', linestyle='--', label=f'Média: {erro.mean():.3f}')
    axes[0, 0].fill_between(x_axis, erro.mean() - erro.std(), erro.mean() + erro.std(),
                            alpha=0.2, color='blue', label='±1σ')
    axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Erro IMU')
    axes[0, 0].set_title('Evolução do Erro IMU'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, delta_erro, 'orange', linewidth=1.2, alpha=0.8)
    axes[0, 1].axhline(0, color='black', alpha=0.5)
    axes[0, 1].axhline(delta_erro.mean(), color='blue', linestyle='--',
                       label=f'Média: {delta_erro.mean():.3f}')
    axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Delta Erro IMU')
    axes[0, 1].set_title('Evolução do Delta Erro IMU'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_axis, uc, 'g-', linewidth=1.2, alpha=0.8)
    axes[1, 0].axhline(0, color='black', alpha=0.5)
    axes[1, 0].axhline(uc.mean(), color='blue', linestyle='--', label=f'Média: {uc.mean():.2f}')
    axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel('UC IMU')
    axes[1, 0].set_title('Evolução do Sinal de Controle IMU'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(curvatura, erro, alpha=0.6, s=8, c=x_axis, cmap='viridis')
    z = np.polyfit(curvatura, erro, 1)
    r = np.corrcoef(curvatura, erro)[0, 1]
    axes[1, 1].plot(curvatura, np.poly1d(z)(curvatura), 'r-', linewidth=2, label=f'r={r:.3f}')
    axes[1, 1].set_xlabel('Curvatura'); axes[1, 1].set_ylabel('Erro IMU')
    axes[1, 1].set_title('Erro vs Curvatura'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 13: Controle IMU - análise espacial (2x2) ─────────────────────────
def plot_controle_imu_espacial(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('CONTROLE IMU - ANÁLISE ESPACIAL', fontsize=14, fontweight='bold')
    x_pos = dados['x']; y_pos = dados['y']
    x_axis = calc['distancia']
    erro = calc['erro_imu']; uc = calc['uc_imu']

    sc1 = axes[0, 0].scatter(x_pos, y_pos, c=erro, cmap='RdBu_r', s=15, alpha=0.8)
    axes[0, 0].set_title('Mapa Erro IMU'); axes[0, 0].set_aspect('equal')
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    fig.colorbar(sc1, ax=axes[0, 0], shrink=0.8).set_label('Erro IMU')

    sc2 = axes[0, 1].scatter(x_pos, y_pos, c=uc, cmap='RdYlGn', s=15, alpha=0.8)
    axes[0, 1].set_title('Mapa UC IMU'); axes[0, 1].set_aspect('equal')
    fig.colorbar(sc2, ax=axes[0, 1], shrink=0.8).set_label('UC IMU')

    axes[1, 0].scatter(x_pos, erro, alpha=0.6, s=8, c=x_axis, cmap='plasma')
    z = np.polyfit(x_pos, erro, 1)
    r = np.corrcoef(x_pos, erro)[0, 1]
    axes[1, 0].plot(x_pos, np.poly1d(z)(x_pos), 'r-', linewidth=2, label=f'r={r:.3f}')
    axes[1, 0].set_xlabel('X (mm)'); axes[1, 0].set_ylabel('Erro IMU')
    axes[1, 0].set_title('Erro IMU vs X'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(y_pos, erro, alpha=0.6, s=8, c=x_axis, cmap='coolwarm')
    z = np.polyfit(y_pos, erro, 1)
    r = np.corrcoef(y_pos, erro)[0, 1]
    axes[1, 1].plot(y_pos, np.poly1d(z)(y_pos), 'r-', linewidth=2, label=f'r={r:.3f}')
    axes[1, 1].set_xlabel('Y (mm)'); axes[1, 1].set_ylabel('Erro IMU')
    axes[1, 1].set_title('Erro IMU vs Y'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 14: Trajetória suavizada com filtros (comparação) ─────────────────
def plot_trajetoria_suavizada(fig, dados, calc):
    ax = fig.add_subplot(111)
    x_suave = calc['x_suave']; y_suave = calc['y_suave']
    trechos = calc['trechos_gaussiano']

    ax.plot(dados['x'], dados['y'], 'b-', alpha=0.4, linewidth=2, label='Trajetória Original')
    ax.plot(x_suave, y_suave, 'r-', alpha=0.9, linewidth=2, label='Trajetória Suavizada')

    for i, trecho in enumerate(trechos):
        inicio = trecho['inicio']; fim = trecho['fim']; sigma = trecho['sigma']
        ax.plot(dados['x'].iloc[inicio:fim + 1], dados['y'].iloc[inicio:fim + 1],
                color='orange', linewidth=4, alpha=0.7, label=f'Trecho Gaussiano {i + 1} (σ={sigma})')

    ax.scatter(x_suave[0], y_suave[0], color='green', s=200, marker='o',
               edgecolors='white', linewidth=3, label='Início', zorder=10)
    ax.scatter(x_suave[-1], y_suave[-1], color='purple', s=200, marker='X',
               edgecolors='white', linewidth=3, label='Fim', zorder=10)

    ax.set_xlabel('Posição X (mm)', fontsize=12); ax.set_ylabel('Posição Y (mm)', fontsize=12)
    ax.set_title('Original vs Suavizada com Filtros Gaussianos', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    fig.tight_layout()


# ── GRÁFICO 15: Perfil de PWM por curvatura (2x2) ─────────────────────────────
def plot_perfil_pwm(fig, dados, calc):
    axes = fig.subplots(2, 2)
    fig.suptitle('PERFIL DE PWM BASEADO EM CURVATURA', fontsize=14, fontweight='bold')
    x_suave = calc['x_suave']; y_suave = calc['y_suave']
    res = calc['resultado_pwm']
    pwm = res['pwm']; curv_norm = res['curvatura_normalizada']; em_curva = res['em_curva']
    pontos = np.arange(len(pwm))

    sc1 = axes[0, 0].scatter(x_suave, y_suave, c=pwm, cmap='RdYlGn', s=15, alpha=0.8)
    axes[0, 0].set_title('Mapa de PWM'); axes[0, 0].set_aspect('equal')
    fig.colorbar(sc1, ax=axes[0, 0], shrink=0.8).set_label('PWM')

    sc2 = axes[0, 1].scatter(x_suave, y_suave, c=curv_norm, cmap='viridis', s=15, alpha=0.8)
    axes[0, 1].set_title('Mapa de Curvatura'); axes[0, 1].set_aspect('equal')
    fig.colorbar(sc2, ax=axes[0, 1], shrink=0.8).set_label('Curvatura Norm.')

    axes[1, 0].plot(pontos, pwm, 'b-', linewidth=1.5, label='PWM Final')
    axes[1, 0].plot(pontos, res['pwm_originais'], 'r--', alpha=0.7, label='PWM Original')
    em_curva_arr = np.array(em_curva)
    mudancas = np.diff(em_curva_arr.astype(int))
    inic_curvas = np.where(mudancas == 1)[0] + 1
    fins_curvas = np.where(mudancas == -1)[0] + 1
    if em_curva_arr[0]:
        inic_curvas = np.concatenate([[0], inic_curvas])
    if em_curva_arr[-1]:
        fins_curvas = np.concatenate([fins_curvas, [len(em_curva_arr)]])
    for ic, fc in zip(inic_curvas, fins_curvas):
        axes[1, 0].axvspan(ic - 0.5, fc - 0.5, alpha=0.15, color='red')
    axes[1, 0].set_xlabel('Ponto da Trajetória'); axes[1, 0].set_ylabel('PWM')
    axes[1, 0].set_title('Perfil de PWM'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(curv_norm, pwm, alpha=0.6, s=15)
    axes[1, 1].set_xlabel('Curvatura Normalizada'); axes[1, 1].set_ylabel('PWM')
    axes[1, 1].set_title('Curvatura vs PWM'); axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()


# ── GRÁFICO 16: Reamostragem final com PWM por cor ────────────────────────────
def plot_reamostragem_final(fig, dados, calc):
    ax = fig.add_subplot(111)
    x_suave = calc['x_suave']; y_suave = calc['y_suave']
    x_dist = calc['x_dist']; y_dist = calc['y_dist']; pwm_dist = calc['pwm_dist']
    distancia_alvo = calc['distancia_alvo']

    ax.plot(dados['x'], dados['y'], 'b-', alpha=0.25, linewidth=15, label='Original')
    ax.plot(x_suave, y_suave, 'r-', alpha=1, linewidth=2, label='Suavizado')
    sc = ax.scatter(x_dist, y_dist, c=pwm_dist, s=50, alpha=1, cmap='RdYlGn',
                    label=f'Pontos Dist={distancia_alvo}mm', zorder=5,
                    edgecolors='black', linewidth=1)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('PWM', fontsize=12, fontweight='bold')

    ax.set_title(f'Reamostragem por Distância ({distancia_alvo}mm) - PWM por Cor',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Posição X (mm)', fontsize=12); ax.set_ylabel('Posição Y (mm)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    fig.tight_layout()


# ==============================================================================
# DEFINIÇÃO DA LISTA DE GRÁFICOS
# ==============================================================================
GRAFICOS = [
    ("Trajetória + Marcas",             plot_trajetoria_marcas,               {}),
    ("Trajetória: Planejada vs Real",   plot_trajetoria_planejada_realizada,  {}),
    ("Encoders Acumulados (2x2)",       plot_encoders_acumulados,             {}),
    ("Pulsos Incrementais (2x2)",       plot_pulsos_incrementais,             {}),
    ("Orientação θ (2x2)",              plot_orientacao_theta,                {}),
    ("Velocidades Lin/Ang (2x2)",       plot_velocidades,                     {}),
    ("Controle Frontal - Temporal",     plot_controle_frontal_temporal,       {}),
    ("Controle Frontal - Correlações",  plot_controle_frontal_correlacoes,    {}),
    ("Duty Cycle Motores (2x2)",        plot_duty_cycle,                      {}),
    ("Sensores Frontais - Heatmap",     plot_sensores_heatmap,                {}),
    ("Sensores Frontais - Linha",       plot_sensores_linha,                  {}),
    ("Controle IMU - Temporal",         plot_controle_imu_temporal,           {}),
    ("Controle IMU - Espacial",         plot_controle_imu_espacial,           {}),
    ("Trajetória Suavizada",            plot_trajetoria_suavizada,            {}),
    ("Perfil de PWM (2x2)",             plot_perfil_pwm,                      {}),
    ("Reamostragem Final + PWM",        plot_reamostragem_final,              {}),
]



class VisualizadorGraficos:
    """
    Arquitetura de dois frames fixos:
      - frame_menu:    sempre existe, mostrado/escondido com pack/pack_forget
      - frame_grafico: sempre existe, mostrado/escondido com pack/pack_forget

    Isso evita destruir e recriar widgets a cada navegação, que era a causa
    do bug gráfico ao voltar ao menu.
    """

    def __init__(self, dados, dados_original, calc):
        self.dados = dados
        self.dados_original = dados_original
        self.calc = calc
        self.todos_graficos = GRAFICOS

        # ── Janela principal ──────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Visualizador de Planejamento")
        self.root.state("zoomed")
        self.root.configure(bg="#202020")

        # ── Frame do menu (fixo, jamais destruído) ────────────────────────────
        self.frame_menu = tk.Frame(self.root, bg="#202020")

        # ── Frame do gráfico (fixo, jamais destruído) ─────────────────────────
        self.frame_grafico = tk.Frame(self.root, bg="#1e1e1e")

        # Figura e canvas matplotlib ativos (trocados a cada gráfico)
        self._fig_ativa = None
        self._canvas_ativo = None
        self.indice_grafico_ativo = None

        # ── Constrói o menu uma única vez ─────────────────────────────────────
        self._construir_menu()

        # Começa mostrando o menu
        self._mostrar_menu()

        self.root.mainloop()

    # =========================================================================
    # CONSTRUÇÃO DO MENU (executado apenas uma vez no __init__)
    # =========================================================================
    def _construir_menu(self):
        """Cria todos os widgets do menu dentro de self.frame_menu."""

        # Título
        tk.Label(
            self.frame_menu,
            text="Visualizador de Planejamento",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#1e1e1e",
            pady=20
        ).pack(fill="x")

        # Barra de pesquisa
        pesquisa_frame = tk.Frame(self.frame_menu, bg="#202020")
        pesquisa_frame.pack(pady=10)

        tk.Label(
            pesquisa_frame,
            text="Pesquisar gráfico:",
            font=("Arial", 13, "bold"),
            fg="white",
            bg="#202020"
        ).pack(side="left", padx=(20, 5))

        self.var_pesquisa = tk.StringVar()
        self.var_pesquisa.trace_add("write", self._filtrar_botoes)

        tk.Entry(
            pesquisa_frame,
            textvariable=self.var_pesquisa,
            font=("Arial", 12),
            width=40
        ).pack(side="left", padx=5)

        tk.Button(
            pesquisa_frame,
            text="📡 Enviar via Wi-Fi",
            font=("Arial", 12, "bold"),
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            activeforeground="white",
            padx=12,
            pady=4,
            relief="raised",
            bd=3,
            command=self._abrir_dialogo_wifi
        ).pack(side="left", padx=(10, 5))

        tk.Button(
            pesquisa_frame,
            text="📥 Receber do ESP32",
            font=("Arial", 12, "bold"),
            bg="#b45309",
            fg="white",
            activebackground="#92400e",
            activeforeground="white",
            padx=12,
            pady=4,
            relief="raised",
            bd=3,
            command=self._abrir_dialogo_receber
        ).pack(side="left", padx=(5, 20))

        # Grade de botões (sem scrollbar — todos cabem na tela)
        self.frame_botoes = tk.Frame(self.frame_menu, bg="#202020")
        self.frame_botoes.pack(fill="both", expand=True, pady=10)

        self.botoes = []
        colunas = 3

        for idx, (titulo, func, kwargs) in enumerate(self.todos_graficos):
            btn = tk.Button(
                self.frame_botoes,
                text=titulo,
                font=("Arial", 12, "bold"),
                bg="#2563eb",
                fg="white",
                activebackground="#1d4ed8",
                activeforeground="white",
                width=35,
                height=3,
                relief="raised",
                bd=4,
                command=lambda i=idx: self._abrir_grafico(i)
            )
            linha = idx // colunas
            coluna = idx % colunas
            btn.grid(row=linha, column=coluna, padx=20, pady=15, sticky="nsew")
            self.botoes.append((btn, titulo.lower()))

        for c in range(colunas):
            self.frame_botoes.grid_columnconfigure(c, weight=1, uniform="col")

        # ── Conteúdo fixo do frame_grafico (topo com botão Voltar) ───────────
        topo = tk.Frame(self.frame_grafico, bg="#1e1e1e")
        topo.pack(fill="x")

        tk.Button(
            topo,
            text="← Voltar",
            font=("Arial", 12, "bold"),
            bg="#dc2626",
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            padx=15,
            pady=8,
            command=self._mostrar_menu
        ).pack(side="left", padx=15, pady=10)

        self.lbl_titulo_grafico = tk.Label(
            topo,
            text="",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        self.lbl_titulo_grafico.pack(side="left", padx=20)

        self.lbl_teclas = tk.Label(
            topo,
            text="← / → para navegar nos gráficos   |   Esc para voltar ao menu",
            font=("Arial", 11),
            fg="white",
            bg="#1e1e1e"
        )
        self.lbl_teclas.pack(side="right", padx=20)

        # Área onde o canvas matplotlib será inserido
        self.frame_canvas = tk.Frame(self.frame_grafico, bg="#1e1e1e")
        self.frame_canvas.pack(fill="both", expand=True)

    # =========================================================================
    # ENVIO VIA WI-FI
    # =========================================================================
    def _abrir_dialogo_wifi(self):
        """Abre uma janela para configurar IP/porta e enviar os dados ao ESP32."""
      

        dialogo = tk.Toplevel(self.root)
        dialogo.title("Enviar Trajetória via Wi-Fi")
        dialogo.configure(bg="#1e1e1e")
        dialogo.resizable(False, False)
        dialogo.grab_set()  # Bloqueia a janela principal enquanto o diálogo está aberto

        # Centraliza o diálogo
        dialogo.geometry("400x260")
        dialogo.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 200
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 130
        dialogo.geometry(f"+{x}+{y}")

        def label(parent, texto):
            tk.Label(parent, text=texto, font=("Arial", 11), fg="white",
                     bg="#1e1e1e", anchor="w").pack(fill="x", padx=30, pady=(10, 0))

        label(dialogo, "Endereço IP do ESP32:")
        var_ip = tk.StringVar(value="192.168.4.1")
        tk.Entry(dialogo, textvariable=var_ip, font=("Arial", 12), width=28).pack(padx=30)

        label(dialogo, "Porta TCP:")
        var_porta = tk.StringVar(value="5000")
        tk.Entry(dialogo, textvariable=var_porta, font=("Arial", 12), width=28).pack(padx=30)

        # Barra de status
        var_status = tk.StringVar(value="")
        lbl_status = tk.Label(dialogo, textvariable=var_status, font=("Arial", 10),
                              fg="#facc15", bg="#1e1e1e")
        lbl_status.pack(pady=(10, 0))

        def ao_enviar():
            ip = var_ip.get().strip()
            try:
                porta = int(var_porta.get().strip())
            except ValueError:
                mb.showerror("Erro", "Porta inválida.", parent=dialogo)
                return

            var_status.set("Conectando...")
            dialogo.update_idletasks()

            sucesso, mensagem = self._enviar_via_wifi(ip, porta)

            if sucesso:
                var_status.set(f"✔ {mensagem}")
                lbl_status.config(fg="#4ade80")
                dialogo.after(2000, dialogo.destroy)
            else:
                var_status.set(f"✖ {mensagem}")
                lbl_status.config(fg="#f87171")

        tk.Button(
            dialogo,
            text="📡 Enviar",
            font=("Arial", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=20,
            pady=6,
            command=ao_enviar
        ).pack(pady=15)

    def _enviar_via_wifi(self, ip, porta):
        """
        Serializa os três arrays e os envia ao ESP32 via TCP.

        Protocolo (little-endian):
          [4 bytes] N          — número de pontos (uint32)
          [N*2 bytes] ref_x    — coordenadas X (int16)
          [N*2 bytes] ref_y    — coordenadas Y (int16)
          [N*2 bytes] ref_pwm  — valores de PWM (int16)
        Total: 4 + N*6 bytes
        """
       

        x_dist   = self.calc['x_dist']
        y_dist   = self.calc['y_dist']
        pwm_dist = self.calc['pwm_dist']
        n = len(x_dist)

        # Monta o pacote binário completo
        pacote  = struct.pack('<I', n)                          # uint32 — quantidade de pontos
        pacote += struct.pack(f'<{n}h', *[int(v) for v in x_dist])    # int16[] ref_x
        pacote += struct.pack(f'<{n}h', *[int(v) for v in y_dist])    # int16[] ref_y
        pacote += struct.pack(f'<{n}h', *[int(v) for v in pwm_dist])  # int16[] ref_pwm

        try:
            with socket.create_connection((ip, porta), timeout=5) as s:
                s.sendall(pacote)

                # Aguarda confirmação do ESP32 (4 bytes: número de pontos recebidos)
                resposta = s.recv(4)
                if len(resposta) == 4:
                    n_confirmado = struct.unpack('<I', resposta)[0]
                    if n_confirmado == n:
                        return True, f"{n} pontos enviados e confirmados pelo ESP32."
                    else:
                        return False, f"ESP32 confirmou apenas {n_confirmado} de {n} pontos."
                else:
                    return False, "ESP32 não enviou confirmação."

        except ConnectionRefusedError:
            return False, "Conexão recusada. ESP32 está ouvindo na porta?"
        except TimeoutError:
            return False, "Timeout. Verifique o IP e se o ESP32 está acessível."
        except OSError as e:
            return False, f"Erro de rede: {e}"

    # =========================================================================
    # RECEBIMENTO CSV VIA WI-FI (STM → ESP32 → Python)
    # =========================================================================
    def _abrir_dialogo_receber(self):
        """Abre janela para receber CSV do STM via ESP32 e carregar direto no app."""

        dialogo = tk.Toplevel(self.root)
        dialogo.title("Receber CSV do STM")
        dialogo.configure(bg="#1e1e1e")
        dialogo.resizable(False, False)
        dialogo.grab_set()

        dialogo.geometry("500x380")
        dialogo.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 190
        dialogo.geometry(f"+{x}+{y}")

        def label(texto):
            tk.Label(dialogo, text=texto, font=("Arial", 11), fg="white",
                     bg="#1e1e1e", anchor="w").pack(fill="x", padx=30, pady=(8, 0))

        label("IP do ESP32:")
        var_ip = tk.StringVar(value="192.168.4.1")
        tk.Entry(dialogo, textvariable=var_ip, font=("Arial", 12), width=32).pack(padx=30)

        label("Porta TCP (CSV do STM):")
        var_porta = tk.StringVar(value="5001")
        tk.Entry(dialogo, textvariable=var_porta, font=("Arial", 12), width=32).pack(padx=30)

        label("Progresso:")
        txt_frame = tk.Frame(dialogo, bg="#1e1e1e")
        txt_frame.pack(fill="both", expand=True, padx=30, pady=(4, 0))
        txt_linhas = tk.Text(txt_frame, height=7, font=("Courier", 9),
                             bg="#111827", fg="#86efac", insertbackground="white")
        scroll = tk.Scrollbar(txt_frame, command=txt_linhas.yview)
        txt_linhas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        txt_linhas.pack(side="left", fill="both", expand=True)

        var_status = tk.StringVar(value="")
        lbl_status = tk.Label(dialogo, textvariable=var_status, font=("Arial", 10),
                              fg="#facc15", bg="#1e1e1e", wraplength=460)
        lbl_status.pack(pady=(6, 0), padx=10)

        btn_receber = tk.Button(dialogo, text="📥 Receber do ESP32",
                                font=("Arial", 12, "bold"), bg="#b45309", fg="white",
                                activebackground="#92400e", activeforeground="white",
                                padx=16, pady=6)
        btn_receber.pack(pady=10)

        def adicionar_linha(linha):
            txt_linhas.insert("end", linha + "\n")
            txt_linhas.see("end")

        def ao_receber():
            ip = var_ip.get().strip()
            try:
                porta = int(var_porta.get().strip())
            except ValueError:
                var_status.set("Porta inválida.")
                return

            btn_receber.config(state="disabled", text="Recebendo...")
            var_status.set(f"Conectando em {ip}:{porta}...")
            txt_linhas.delete("1.0", "end")
            lbl_status.config(fg="#facc15")
            dialogo.update_idletasks()

            def tarefa():
                sucesso, resultado = self._receber_csv_wifi(
                    ip, porta,
                    callback=lambda l: dialogo.after(0, adicionar_linha, l)
                )
                if sucesso:
                    linhas = resultado
                    # Carrega as linhas como DataFrame e atualiza o app
                    ok, msg = self._carregar_csv_das_linhas(linhas)
                    if ok:
                        dialogo.after(0, lambda: var_status.set(
                            f"✔ {len(linhas)} linhas recebidas e carregadas!\nGráficos prontos."))
                        dialogo.after(0, lambda: lbl_status.config(fg="#4ade80"))
                        dialogo.after(2000, dialogo.destroy)
                    else:
                        dialogo.after(0, lambda: var_status.set(f"✔ Recebido, mas erro ao carregar: {msg}"))
                        dialogo.after(0, lambda: lbl_status.config(fg="#f87171"))
                else:
                    dialogo.after(0, lambda: var_status.set(f"✖ {resultado}"))
                    dialogo.after(0, lambda: lbl_status.config(fg="#f87171"))
                dialogo.after(0, lambda: btn_receber.config(
                    state="normal", text="📥 Receber do ESP32"))

            threading.Thread(target=tarefa, daemon=True).start()

        btn_receber.config(command=ao_receber)

    def _carregar_csv_das_linhas(self, linhas):
        """
        Converte as linhas CSV recebidas em DataFrame e atualiza
        self.dados, self.dados_original e self.calc — exatamente como
        se o arquivo tivesse sido lido do disco.
        """
        try:
            # Filtra marcadores do protocolo BT
            linhas_csv = [l for l in linhas
                          if not l.startswith('BT_START') and not l.startswith('BT_END:')]

            texto = '\n'.join(linhas_csv)
            dados_original = pd.read_csv(io.StringIO(texto), sep=SEPARADOR_CSV)
            dados = pd.read_csv(io.StringIO(texto), sep=SEPARADOR_CSV)

            # Mesmo pré-processamento de carregar_dados()
            dados['y'] = dados['y'] - dados['y'].iloc[0]
            dados['θ'] = dados['θ'] - dados['θ'].iloc[0]

            calc = preparar_dados(dados, dados_original)

            # Atualiza o estado do app
            self.dados          = dados
            self.dados_original = dados_original
            self.calc           = calc

            return True, f"{len(dados)} linhas carregadas."
        except Exception as e:
            return False, str(e)

    def _receber_csv_wifi(self, ip, porta=5001, callback=None):
     

        linhas = []
        try:
            with socket.create_connection((ip, porta), timeout=30) as s:
                s.settimeout(30)
                buf = b''
                while True:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        linha_bytes, buf = buf.split(b'\n', 1)
                        linha = linha_bytes.decode('utf-8', errors='replace').strip()
                        if not linha:
                            continue
                        linhas.append(linha)
                        if callback:
                            callback(linha)
                        if linha.startswith('BT_END:'):
                            return True, linhas

            return True, linhas

        except TimeoutError:
            if linhas:
                return True, linhas
            return False, "Timeout. ESP32 não enviou dados em 30 segundos."
        except ConnectionRefusedError:
            return False, "Conexão recusada. ESP32 está ouvindo na porta 5001?"
        except OSError as e:
            return False, f"Erro de rede: {e}"

    def _recv_all(self, conn, n_bytes):
        """Garante receber exatamente n_bytes do socket."""
        buf = b''
        while len(buf) < n_bytes:
            chunk = conn.recv(n_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    # =========================================================================
    # ALTERNÂNCIA MENU ↔ GRÁFICO
    # =========================================================================
    def _mostrar_menu(self):
        """Esconde o frame do gráfico e mostra o menu."""
        # Libera figura/canvas antes de sair da tela do gráfico
        self._destruir_canvas()
        self._unbind_teclas()
        self.indice_grafico_ativo = None
        self.frame_grafico.pack_forget()
        self.frame_menu.pack(fill="both", expand=True)

    def _mostrar_grafico(self):
        """Esconde o menu e mostra o frame do gráfico."""
        self.frame_menu.pack_forget()
        self.frame_grafico.pack(fill="both", expand=True)
        self._bind_teclas()

    # =========================================================================
    # GERENCIAMENTO DO CANVAS MATPLOTLIB
    # =========================================================================
    def _destruir_canvas(self):
        """Remove o canvas matplotlib atual e fecha a figura, sem tocar nos
        outros widgets de frame_grafico (botão Voltar, título, etc.)."""
        if self._canvas_ativo is not None:
            self._canvas_ativo.get_tk_widget().destroy()
            self._canvas_ativo = None
        if self._fig_ativa is not None:
            plt.close(self._fig_ativa)
            self._fig_ativa = None

    def _bind_teclas(self):
        """Ativa atalhos de teclado para o modo gráfico."""
        self.root.bind('<Left>', self._on_seta_esquerda)
        self.root.bind('<Right>', self._on_seta_direita)
        self.root.bind('<Escape>', self._on_esc)

    def _unbind_teclas(self):
        """Desativa atalhos de teclado ao retornar ao menu."""
        self.root.unbind('<Left>')
        self.root.unbind('<Right>')
        self.root.unbind('<Escape>')

    def _on_seta_esquerda(self, event=None):
        if self.indice_grafico_ativo is not None:
            self._navegar_grafico(-1)

    def _on_seta_direita(self, event=None):
        if self.indice_grafico_ativo is not None:
            self._navegar_grafico(1)

    def _on_esc(self, event=None):
        self._mostrar_menu()

    def _navegar_grafico(self, delta):
        if self.indice_grafico_ativo is None:
            return
        novo_indice = (self.indice_grafico_ativo + delta) % len(self.todos_graficos)
        self._abrir_grafico(novo_indice)

    # =========================================================================
    # PESQUISA
    # =========================================================================
    def _filtrar_botoes(self, *args):
        termo = self.var_pesquisa.get().lower().strip()
        colunas = 3
        for idx, (btn, titulo) in enumerate(self.botoes):
            if termo in titulo:
                btn.grid(row=idx // colunas, column=idx % colunas,
                         padx=20, pady=15, sticky="nsew")
            else:
                btn.grid_remove()

    # =========================================================================
    # ABRIR GRÁFICO
    # =========================================================================
    def _abrir_grafico(self, indice):
        """Renderiza o gráfico escolhido dentro de frame_canvas."""
        # Verifica se os dados já foram carregados via Wi-Fi
        if self.dados is None:
            mb.showwarning(
                "Sem dados",
                "Nenhum dado carregado ainda.\n\nClique em '📥 Receber do ESP32' para receber o CSV do STM.",
                parent=self.root
            )
            return

        # Destrói canvas anterior (se houver) antes de criar o novo
        self._destruir_canvas()

        titulo, func, kwargs = self.todos_graficos[indice]
        self.lbl_titulo_grafico.config(text=titulo)

        # Usa o tamanho atual do frame para deixar o gráfico estendido corretamente
        self.root.update_idletasks()
        largura = max(16, self.frame_canvas.winfo_width() / 100)
        altura = max(9, self.frame_canvas.winfo_height() / 100)
        fig = plt.Figure(figsize=(largura, altura), dpi=100)
        self._fig_ativa = fig

        try:
            params = inspect.signature(func).parameters
            args = [fig]
            if 'dados' in params:
                args.append(self.dados)
            if 'dados_original' in params:
                args.append(self.dados_original)
            if 'calc' in params:
                args.append(self.calc)
            func(*args, **kwargs)

        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f"Erro ao renderizar gráfico:\n{e}\n\n{traceback.format_exc()}",
                    ha="center", va="center", fontsize=11, color="red",
                    transform=ax.transAxes, wrap=True)
            ax.axis("off")

        # Insere canvas dentro de frame_canvas (filho fixo de frame_grafico)
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        self._canvas_ativo = canvas
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        self.root.update_idletasks()
        canvas.draw_idle()
        widget.focus_set()

        self.indice_grafico_ativo = indice
        # Troca de tela
        self._mostrar_grafico()


# ==============================================================================
# PONTO DE ENTRADA

# ==============================================================================
if __name__ == '__main__':
    
    #VisualizadorGraficos(dados=None, dados_original=None, calc=None)
    try:
       dados, dados_original, calc, csv_usado = carregar_csv_de_teste(caminho_csv)
       print(f"Usando CSV de teste: {csv_usado}")
       VisualizadorGraficos(dados=dados, dados_original=dados_original, calc=calc)
    except Exception as e:
       print(f"Não foi possível carregar CSV de teste: {e}")
       print("Iniciando o app sem dados. Use '📥 Receber do ESP32' para carregar um CSV via Wi-Fi.")
       VisualizadorGraficos(dados=None, dados_original=None, calc=None)