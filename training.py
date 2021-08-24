import random
import numpy 
import Library2
import math
import ast
import copy
from PIL import Image
import numpy as np


#Ввод массива из картинок
########################################################################################################## 
data = [] #Массив входных картинок

for m in range(0,10): #Проходим по всем цифрам
    for i in range (1,8): #Проходим по всем номерам цифр
        img = Image.open('images/' + str(m) + '_' + str(i) + '.png') #Открываем картинку с нужным названием
        gray = img.convert('L')
        bw = gray.point(lambda x: 0 if x<128 else 1, '1') #Преобразуем
        data.append(list(bw.getdata())) #Добавляем ее в массив
#####################################################################################################

#Ручной ввод#
################################################################################
quantity_Output_Neirons = 1 #Количество выходных неиронов
quantity_Iterations = 70 #Количество итераций
quantity_Input_Neirons = 247 #Количество входных неиронов 247
quantity_Hiden_Layers = 2
speed = 0.1 #Переменная скорости обучения
moment = 0.07 #Переменная момента
epoh = 50 #Количество эпох
################################################################################

#Создание массивов
########################################################################################################################################
iteration_Inputs = data #Ввод массивов картинок
#ideal_Results = Library2.ideal_Result_Input(quantity_Iterations,quantity_Output_Neirons) #Ручной ввод идеальных результатов
hiden_Neirons = Library2.add_Hiden_Neirons(quantity_Hiden_Layers) #Создание массива скрытых неиронов
sinopses = Library2.addSinops(quantity_Input_Neirons,hiden_Neirons,quantity_Output_Neirons) #Создание массива синопсов
sinopses = Library2.arrange_Random_Values_Sinopses(sinopses) #Заполнение массива синопсов случайными значениями от -1 до 1
hiden_Delts = Library2.add_Hiden_Delts(hiden_Neirons) #Создание массива дельт скрытых неиронов
sinopses_Change = Library2.add_Sinopses_Change(sinopses) #Создание массива изменения значения синопсов
########################################################################################################################################

if int(input('Загрузить значения синопсов и неиронов из файла? ')) == 1: #Загрузка синопсов и неиронов из файла(Возможно только при одинаковом количестве неиронов на всех слоях)
    hiden_Neirons = [0]*quantity_Hiden_Layers
    sinopses = np.loadtxt('sinopses_' + str(int(input('Введите номер сохранения: '))) + '.txt') #Загрузка с номером цифры
    for counter in range(quantity_Hiden_Layers):
        hiden_Neirons[counter] = np.loadtxt('hiden_Neirons_' + str(int(input('Введите номер сохранения: '))) + '_' + str(counter)+'.txt') #Загрузка с номером цифры

ideal_Results = [[0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [0],[0],[0],[0],[0],[0],[0],
                 [1],[1],[1],[1],[1],[1],[1]]  #Создание массива идеальных результатов(для ускорения работы)

############################################################################################# Обучение сети              
for epoh_Counter in range(epoh): #Повтор по количеству эпох
    for iteration_Number in range(quantity_Iterations): #Прохождение по всем итерациям
        final_Neiron = Library2.neuroNet_Output_Calculation(iteration_Inputs,sinopses,quantity_Input_Neirons,hiden_Neirons,iteration_Number,quantity_Hiden_Layers,quantity_Output_Neirons) #Расчет значения финальных неиронов
        hiden_Neirons = final_Neiron[1] #Перераспределение на скрытые нейроны
        final_Neiron = final_Neiron[0] #Перераспределение на выходные нейроны
        print(Library2.error_Calculation(final_Neiron,ideal_Results,iteration_Number), final_Neiron, iteration_Number) #Вывод ошибки конечных неиронов и номера итерации(Закоментировать для ускорения расчетов)
        output_Delta = Library2.output_Neiron_Delta_Calculation(final_Neiron,ideal_Results,iteration_Number,quantity_Output_Neirons) #Расчет дельт выходных неиронов
        hiden_Neirons_New = copy.deepcopy(hiden_Neirons) #Создание копии массива для устранения ошибок
        hiden_Delts = Library2.hiden_Delts_Calculation(hiden_Neirons_New,sinopses,output_Delta,quantity_Hiden_Layers,hiden_Delts,quantity_Output_Neirons) #Расчет дельт скрытых неиронов
        gradients = Library2.sinopses_Grad_Calculation(sinopses,iteration_Number,iteration_Inputs,hiden_Delts,quantity_Input_Neirons,output_Delta,hiden_Neirons_New,quantity_Hiden_Layers,quantity_Output_Neirons) #Расчет градиентов синопсов    
        sinopses_Change = Library2.sinopses_Update(speed,moment,gradients,sinopses_Change,sinopses) #Расчет изменения синопсов и новых синопсов
        sinopses = sinopses_Change[1] #Перераспределение на синопсы
        sinopses_Change = sinopses_Change[0] #Перераспределение на изменение синопсов
    print(epoh_Counter) #Вывод номера эпохи
#############################################################################################


############################################## Сохранение синопсов в файл
if int(input('Сохранить значения синопсов и неиронов? ')) == 1: #Сохранение массивов синопсов и неиронов в файл
    np.savetxt('sinopses_' + str(int(input('Введите номер сохранения: '))) + '.txt',sinopses) #Сохранение с вводом цифры
    for counter in range(len(hiden_Neirons)):
        np.savetxt('hiden_Neirons_' + str(int(input('Введите номер сохранения: '))) +'_'+str(counter)+'.txt',hiden_Neirons[counter]) #Сохранение с номером цифры
##############################################


exit=str(input('Нажмите любую клавишу чтобы выйти')) #Выход из файла
