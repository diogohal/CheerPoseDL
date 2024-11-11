from ultralytics import YOLO
import os
import cv2
import numpy as np
import math
import pandas as pd

joint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
connections = [
    (6,8), (8,10), (5,7), (7,9),         # Braços
    (5,6), (12,11), (6,12), (5,11),      # Tronco
    (12,14), (14,16), (11,13), (13,15)   # Pernas 
]

def calculateAngle(joint1, joint2, joint3):
    """
    Calculate the angle formed by three points in 2D space.
    
    Parameters:
    joint1, joint2, joint3: Tuples representing the coordinates of the points (x, y).
    
    Returns:
    The angle in degrees between the three points.
    """
    # Vectors from joint2 to joint1 and joint2 to joint3
    v1 = (joint1[0] - joint2[0], joint1[1] - joint2[1])
    v2 = (joint3[0] - joint2[0], joint3[1] - joint2[1])

    # Dot product of v1 and v2
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Magnitudes of v1 and v2
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # Cosine of the angle
    cos_theta = dot_product / (mag_v1 * mag_v2)

    # Clamp cosine value to avoid floating point errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # Calculate the angle in radians
    theta_rad = math.acos(cos_theta)

    # Convert the angle to degrees
    theta_deg = math.degrees(theta_rad)

    return int(theta_deg)
    

def calculateDistance(joint1, joint2):
    """
    Calculate the Euclidean distance between two points in 2D space.
    
    Parameters:
    joint1, joint2: Tuples representing the coordinates of the points (x, y).
    
    Returns:
    The Euclidean distance between the two points.
    """
    # Calculate the difference in x and y coordinates
    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]

    # Calculate the Euclidean distance
    distance = math.sqrt(delta_x**2 + delta_y**2)

    return distance

font_size = 2 

# ------------ Criar Dataframe ------------
columns = [
    'cotovelo_direito', 'cotovelo_esquerdo',
    'ombro_direito', 'ombro_esquerdo',
    'cintura_direita', 'cintura_esquerda',
    'joelho_direito', 'joelho_esquerdo',
    'braco_direito', 'braco_esquerdo',
    'figure', 'image_name'
]
df = pd.DataFrame(columns=columns)

# ------------ Extrai informações e desenha a figura ------------
model = YOLO("yolo11l-pose.pt")  # load an official model
figures = ['lib', 'arabesque', 'bow', 'heel', 'scale', 'scorpion']
for figure in figures:
    folder_path = f'Dataset/{figure}'
    for image_name in os.listdir(folder_path):
        print(image_name)
        if not image_name.endswith(('.jpg', '.jpeg', '.png')): 
            continue
        source = os.path.join(folder_path, image_name)
        results = model(source=source, conf=0.3, save=True)
        img = cv2.imread(source)
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        keypoints = results[0].keypoints
        # Assuming keypoints is a list of detected keypoints for each person, we will extract specific joints
        if len(keypoints) > 0:
            for person in keypoints:
                # Extract the keypoints for this person
                xy = person.xy[0]  # (17, 2) array: x, y coordinates
                conf = person.conf[0]  # (17,) array: confidence values

                joint_positions = {}
                
                for i, (x, y) in enumerate(xy):
                    x = x.item()  # Convert tensor to scalar
                    y = y.item()  # Convert tensor to scalar
                    c = conf[i].item()  # Confidence value

                    if c > 0.5:  # Only draw keypoints with confidence > 0.5
                        joint_name = joint_names[i]  # Get the joint name based on the index
                        joint_positions[i] = (int(x), int(y))  # Store the joint's position
                        if(i < 5):
                            continue
                        # Draw the joint on the image
                        # cv2.circle(img, (int(x), int(y)), 20, (0, 255, 0), -1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = str(i)  # Convert the index to a string to display
                        cv2.putText(img, text, (int(x) + 10, int(y) - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw the connections between the joints
                for i, (start_idx, end_idx) in enumerate(connections):
                    if start_idx in joint_positions and end_idx in joint_positions:
                        start_point = joint_positions[start_idx]
                        end_point = joint_positions[end_idx]
                        if(i//4 == 0):
                            cv2.line(img, start_point, end_point, (0, 0, 255), 3)  # Draw the line (blue color)
                        elif(i//4 == 1):
                            cv2.line(img, start_point, end_point, (0, 255, 0), 3)  # Draw the line (green color)
                        elif(i//4 == 2):
                            cv2.line(img, start_point, end_point, (255, 0, 0), 3)  # Draw the line (red color)
                break

        # ------------ Obter Dados ------------
        try:
            joint1, joint2, joint3 = joint_positions[5], joint_positions[7], joint_positions[9]
            cotovelo_direito = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(cotovelo_direito), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Cotovelo direito: {cotovelo_direito}')
        except:
            continue
        
        try:
            joint1, joint2, joint3 = joint_positions[10], joint_positions[8], joint_positions[6]
            cotovelo_esquerdo = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(cotovelo_esquerdo), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Cotovelo esquerdo: {cotovelo_esquerdo}')
        except:
            continue

        try:
            joint1, joint2, joint3 = joint_positions[7], joint_positions[5], joint_positions[11]
            ombro_direito = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(ombro_direito), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Ombro direito: {ombro_direito}')
        except:
            continue

        try:
            joint1, joint2, joint3 = joint_positions[12], joint_positions[6], joint_positions[8]
            ombro_esquerdo = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(ombro_esquerdo), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Ombro esquerdo: {ombro_esquerdo}')
        except:
            continue
        
        try:
            joint1, joint2, joint3 = joint_positions[5], joint_positions[11], joint_positions[13]
            cintura_direita = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(cintura_direita), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Cintura direita: {cintura_direita}')
        except:
            continue
        
        try:
            joint1, joint2, joint3 = joint_positions[6], joint_positions[12], joint_positions[14]
            cintura_esquerda = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(cintura_esquerda), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Cintura esquerda: {cintura_esquerda}')
        except:
            continue
        
        try:
            joint1, joint2, joint3 = joint_positions[15], joint_positions[13], joint_positions[11]
            joelho_direito = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(joelho_direito), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Joelho direito: {joelho_direito}')

        except:
            continue
        
        try:
            joint1, joint2, joint3 = joint_positions[12], joint_positions[14], joint_positions[16]
            joelho_esquerdo = calculateAngle(joint1, joint2, joint3)
            cv2.putText(img, str(joelho_esquerdo), (int(joint2[0]), int(joint2[1])), font, font_size, (255, 255, 255), 3, cv2.LINE_AA)
            print(f'Joelho esquerdo: {joelho_esquerdo}')
        except:
            continue
        
        try:
            joint1, joint2 = joint_positions[12], joint_positions[6]
            tronco = int(calculateDistance(joint1, joint2))
        except:
            continue
        
        try:
            joint1, joint2 = joint_positions[5], joint_positions[9]
            braco_direito = int(calculateDistance(joint1, joint2))
            braco_direito_prop = braco_direito / (tronco + braco_direito)
        except:
            braco_direito = 0
            braco_direito_prop = 0
        try:
            joint1, joint2 = joint_positions[10], joint_positions[6]
            braco_esquerdo = int(calculateDistance(joint1, joint2))
            braco_esquerdo_prop = braco_esquerdo / (tronco + braco_esquerdo)
        except:
            braco_esquerdo = 0
            braco_esquerdo_prop = 0

        print(f'Proporção braço direito/tronco: {braco_direito_prop}')
        print(f'Proporção braço esquerdo/tronco: {braco_esquerdo_prop}')

        # ------------ Preencher Dataframe ------------
        row_data = {
            'cotovelo_direito': cotovelo_direito,
            'cotovelo_esquerdo': cotovelo_esquerdo,
            'ombro_direito': ombro_direito,
            'ombro_esquerdo': ombro_esquerdo,
            'cintura_direita': cintura_direita,
            'cintura_esquerda': cintura_esquerda,
            'joelho_direito': joelho_direito,
            'joelho_esquerdo': joelho_esquerdo,
            'braco_direito': braco_direito_prop,
            'braco_esquerdo': braco_esquerdo_prop,
            'figure': figure,
            'image_name': image_name
        }
        df.loc[len(df)] = row_data

        # print(df)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('output.jpeg', img_rgb)
        df.to_csv('output.csv', sep=',', index=False)
