import cv2
import os

image_count = 0

def extract_frames(video_path, output_folder, frame_interval, figure):
    global image_count
    # Verifica se a pasta de saída existe, senão cria
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carrega o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # Verifica se o vídeo foi carregado corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Loop para capturar e salvar frames
    while True:
        ret, frame = cap.read()

        # Sai do loop se o vídeo terminar
        if not ret:
            break

        # Salva o frame a cada `frame_interval`
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{figure}_{image_count}.jpeg")
            cv2.imwrite(frame_filename, frame)
            print(f"Salvando {frame_filename}")
            image_count += 1

        frame_count += 1

    # Libera o vídeo
    cap.release()
    print("Extração de frames concluída.")

# Exemplo de uso
# video_path = "CheerPose/arabesque/arabesque_maeda.mp4"          # Caminho do vídeo
# output_folder = "frames"                 # Pasta para salvar as imagens
frame_interval = 15                      # Intervalo de frames
# figure = "arabesque"

# extract_frames(video_path, output_folder, frame_interval, figure)

figure = 'scorpion'
folder_path = f'CheerPose/{figure}'
for video in os.listdir(folder_path):
    print(video)
    video_path = os.path.join(folder_path, video)
    output_folder = os.path.join('Dataset', figure)
    extract_frames(video_path, output_folder, frame_interval, figure)
