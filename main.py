import random
import numpy 
import Library2
import math
import ast
import copy
from PIL import Image
import numpy as np


#Загрузка картинки для обработки
######################################################################################## 
img = Image.open('Testing.png') #Загрузка тестового изображения
gray = img.convert('L')
bw = gray.point(lambda x: 0 if x<128 else 1, '1')
dataTesting = list(bw.getdata())  # Вместо list можно использовать другой тип данных
########################################################################################

#Задание вручную начальных параметров нейросети
###############################################################################################
quantity_Input_Neirons = 247 #Задание количества входных неиронов(Зависит от количества пикселей тестового изображения)
quantity_Hiden_Layers = 2 #Задание количества скрытых слоев
quantity_Output_Neirons = 1 #Задание количества выходных слоев
###############################################################################################

#Загрузка данных для всех цифр
###############################################################################################
if int(input('Начать просчет нейросети? (1 - да) ')) == 1: #Загрузка синопсов и неиронов из файла(Возможно только при одинаковом количестве неиронов на всех слоях)
    data_Neirons = [] 
    data_Sinopses  = []
    for count in range(10): #Загрузка данных для всех цифр
        hiden_Neirons = [0]*quantity_Hiden_Layers
        sinopses = np.loadtxt('data/sinopses_' + str(count) + '.txt') #Загрузка с номером цифры
        for counter in range(quantity_Hiden_Layers): #Загрузка данных нейронов для всех слоев
            hiden_Neirons[counter] = np.loadtxt('data/hiden_Neirons_' + str(count) + '_' + str(counter)+'.txt') #Загрузка с номером цифры
        data_Neirons.append(hiden_Neirons)
        data_Sinopses.append(sinopses)
################################################################################################

#Просчет нейросети для всех нейронов
################################################################################
max_Probability = 0
for counter in range(10):
    probability = Library2.neuroNet_Output_Calculation([dataTesting],data_Sinopses[counter],quantity_Input_Neirons,data_Neirons[counter],0,quantity_Hiden_Layers,quantity_Output_Neirons) #Расчет результата    
    if probability[0][0] > max_Probability:
        max_Probability = probability[0][0]
        max_Number = counter

    print('Вероятность цифры ',counter,' равна ',probability[0][0]*100,'%')
################################################################################
        
print('Наиболее вероятно: ',max_Number,' с вероятностью ',max_Probability*100,'%')

exit=str(input('Нажмите любую клавишу чтобы выйти')) #Выход из файла
