import random
import numpy 
import math
import ast


def addIterationInput(quantity_Iterations,quantity_Input_Neirons):
    iteration_Inputs = numpy.array(range(quantity_Iterations*quantity_Input_Neirons)) #Задание массива нужной длины
    iteration_Inputs.shape = (quantity_Iterations,quantity_Input_Neirons) # Задание формы двумерного массива
    for count in range(quantity_Iterations): #Заполнение массива данными вручную
        for count2 in range(quantity_Input_Neirons):
            iteration_Inputs[count][count2] = int(input())
    return iteration_Inputs


def addSinops(quantity_Input_Neirons,hiden_Neirons,quantity_Output_Neirons):
    result = quantity_Input_Neirons*len(hiden_Neirons[0]) #Доп переменная
    for count in range(len(hiden_Neirons)-1):
        result = result + len(hiden_Neirons[count])*len(hiden_Neirons[count+1]) #Расчет количества синопсов
    result = result + len(hiden_Neirons[len(hiden_Neirons)-1])*quantity_Output_Neirons #Расчет количества синопсов
    sinopses = [0] * result #Создание массива
    return sinopses


def arrange_Random_Values_Sinopses(sinopses):
    for count in range(len(sinopses)): #Заполнение весов синопсов случайными значениями от -1 до 1
        sinopses[count] = random.randint(-100,100)*0.01
    return sinopses


def ideal_Result_Input(quantity_Iterations,quantity_Output_Neirons):
    ideal_Results = numpy.array(range(quantity_Iterations*quantity_Output_Neirons))
    ideal_Results.shape = (quantity_Iterations,quantity_Output_Neirons) # Задание формы двумерного массива
    for count in range(quantity_Iterations): #Заполнение массива ид результатов вручную
        for count2 in range(quantity_Output_Neirons):
            ideal_Results[count][count2] = float(input('Вводите идеальные результаты по размеру '+str(+quantity_Iterations)+' '+str(quantity_Output_Neirons)+': '))
    return ideal_Results


def add_Hiden_Neirons(quantity_Hiden_Layers):
    hiden_Neirons = [0] * quantity_Hiden_Layers #Создание массива
    for count in range(quantity_Hiden_Layers):
        inpac = input("Введите количество нейронов в слое: ") #Задание переменных вручную
        hiden_Neirons[count] = [0] * int(inpac)
    return(hiden_Neirons) 


def add_Hiden_Delts(hiden_Neirons):
    hiden_Delts = hiden_Neirons #Создание массива и заполнение 0
    for count in range(len(hiden_Neirons)):
        for count2 in range(len(hiden_Neirons[count])):
            hiden_Delts[count][count2] = 0
    return hiden_Delts        

def neuroNet_Output_Calculation(iteration_Inputs,sinopses,quantity_Input_Neirons,hiden_Neirons,iteration_Number,quantity_Hiden_Layers,quantity_Output_Neirons):
    for count in range(len(hiden_Neirons[0])): #Расчет значения первого скрытого слоя неиронов(Работает при любом количестве входных и скрытых неиронов)
        for count2 in range(quantity_Input_Neirons):
            hiden_Neirons[0][count] = (hiden_Neirons[0][count] + sinopses[count+(count2*len(hiden_Neirons[0]))]*iteration_Inputs[iteration_Number][count2])
            last_Sinops = count+(count2*len(hiden_Neirons[0]))

    for count in range(len(hiden_Neirons[0])): #Функция активации для первого скрытого слоя
        hiden_Neirons[0][count] = 1/(1+(2.7182818284**((-1)*hiden_Neirons[0][count])))
    final_Neiron = [0]*quantity_Output_Neirons

    for count in range(quantity_Hiden_Layers-1): #Расчет нейронов других скрытых слоев
        for count2 in range(len(hiden_Neirons[count+1])):
            for count3 in range(len(hiden_Neirons[count])):
                hiden_Neirons[count+1][count2] = hiden_Neirons[count+1][count2] + hiden_Neirons[count][count3]*sinopses[last_Sinops+1+count2+(count3*len(hiden_Neirons[count+1]))]
        last_Sinops = last_Sinops+1+count2+(count3*len(hiden_Neirons[count+1]))

    for count in range(quantity_Hiden_Layers-1): #Расчет функции активации для других скрытых слоев
        for count2 in range(len(hiden_Neirons[count+1])):
            hiden_Neirons[count+1][count2] = 1/(1+(2.7182818284**((-1)*hiden_Neirons[count+1][count2])))

    for count in range(len(hiden_Neirons[quantity_Hiden_Layers-1])): #Расчет выходных неиронов
        for count2 in range(quantity_Output_Neirons):
            final_Neiron[count2] = final_Neiron[count2] + hiden_Neirons[quantity_Hiden_Layers-1][count]*sinopses[len(sinopses)-quantity_Output_Neirons*len(hiden_Neirons[quantity_Hiden_Layers-1])+count2+count*quantity_Output_Neirons]
    
    for count in range(quantity_Output_Neirons): #Функция активации для выходных неиронов
        final_Neiron[count] = 1/(1+(2.7182818284**((-1)*final_Neiron[count]))) 
    return final_Neiron,hiden_Neirons
def error_Calculation(actual_Result,ideal_Results,iteration_Number):
    error_Num = 0
    for count in range(len(actual_Result)): #Расчет ошибки финального результата
        error_Num = error_Num + (ideal_Results[iteration_Number][count]-actual_Result[count])**2 
        counter = count
    error_Num = error_Num/(count+1) #Расчет ошибки финального результата
    return error_Num

def output_Neiron_Delta_Calculation(actual_Result,ideal_Results,iteration_Number,quantity_Output_Neirons):
    output_Delta = [0]*quantity_Output_Neirons
    for count in range(quantity_Output_Neirons): #Расчет значения дельты выходного неирона
        output_Delta[count] = (ideal_Results[iteration_Number][count]-actual_Result[count])*(1-actual_Result[count])*actual_Result[count] 
    return output_Delta

def hiden_Delts_Calculation(hiden_Neirons,sinopses,output_Delta,quantity_Hiden_Layers,hiden_Delts,quantity_Output_Neirons):
    delt_Summ = 0
    for count in range(len(hiden_Neirons[quantity_Hiden_Layers-1])): #Расчет дельт последнего слоя
        for count2 in range(quantity_Output_Neirons):
            delt_Summ = delt_Summ + sinopses[len(sinopses) - len(hiden_Neirons[quantity_Hiden_Layers - 1])*quantity_Output_Neirons + count2 + quantity_Output_Neirons*count] * output_Delta[count2] #Расчет суммы дельты к неирону
        hiden_Delts[quantity_Hiden_Layers-1][count] = delt_Summ * hiden_Neirons[quantity_Hiden_Layers-1][count] * (1 - hiden_Neirons[quantity_Hiden_Layers-1][count]) 
        delt_Summ = 0     
    
    last_Sinops = len(sinopses)-len(hiden_Neirons[quantity_Hiden_Layers-1]) * quantity_Hiden_Layers #Расчет номера последнего использованного синопса
    delt_Summ = 0
    
    for count in range(quantity_Hiden_Layers-1): #Цикл по скрытым слоям
        for count2 in range(len(hiden_Neirons[quantity_Hiden_Layers-2-count])): #Цикл по неиронам в слое
            for count3 in range(len(hiden_Neirons[quantity_Hiden_Layers-1-count])): #Цикл по расчету сумм произведений синопсов на приходящий неирон
                #Расчет суммы произведений синопсов на приходящий неирон
                delt_Summ = delt_Summ + hiden_Delts[quantity_Hiden_Layers-1-count][count3]*sinopses[last_Sinops-len(hiden_Neirons[quantity_Hiden_Layers-2-count])*len(hiden_Neirons[quantity_Hiden_Layers-1-count])+count3+count2*len(hiden_Neirons[quantity_Hiden_Layers-1-count])]

            #Расчет значения дельты определенного скрытого неирона
            hiden_Delts[quantity_Hiden_Layers-2-count][count2] = delt_Summ * hiden_Neirons[quantity_Hiden_Layers-2-count][count2] * (1 - hiden_Neirons[quantity_Hiden_Layers-2-count][count2])
            delt_Summ = 0 

        last_Sinops = last_Sinops - len(hiden_Neirons[quantity_Hiden_Layers-1-count])*len(hiden_Neirons[quantity_Hiden_Layers-2-count]) #Обновление номера последнего синопса на первый синопс между двумя последнеми использованными слоями
    return hiden_Delts

def add_Sinopses_Change(sinopses):
    sinopses_Change = [0]*len(sinopses) #Добавление массива изменения синопсов
    return sinopses_Change

def sinopses_Grad_Calculation(sinopses,iteration_Number,iteration_Inputs,hiden_Delts,quantity_Input_Neirons,output_Delta,hiden_Neirons,quantity_Hiden_Layers,quantity_Output_Neirons):
    gradients = [0]*len(sinopses) #Создание массива градиентов синопсов
    for count in range(quantity_Input_Neirons): #Расчет градиентов синопсов между входным и скрытым слоем
        for count2 in range(len(hiden_Delts[0])):
            gradients[count2+count*len(hiden_Delts[0])] = iteration_Inputs[iteration_Number][count] * hiden_Delts[0][count2]
                  
    last_Sinops = count2+count*len(hiden_Delts[0])+1 #Расчет индекса последнего использованного синопса
    
    for count in range(quantity_Hiden_Layers): #Расчет значений градиентов синопсов между скрытыми слоями
        for count2 in range(len(hiden_Delts[count])):
            if count != quantity_Hiden_Layers-1:
                for count3 in range(len(hiden_Delts[count+1])):
                    gradients[(count*len(hiden_Delts[0]))+count3+count2*len(hiden_Delts[count+1])+last_Sinops] = hiden_Neirons[count][count2] * hiden_Delts[count+1][count3]

    for count in range (len(hiden_Neirons[quantity_Hiden_Layers-1])): #Расчет градиентов синопсов между последним скрытым и выходным неироном  
        for count2 in range(quantity_Output_Neirons):
            gradients[len(gradients)-quantity_Output_Neirons*len(hiden_Neirons[quantity_Hiden_Layers-1])+count2+count*quantity_Output_Neirons] = hiden_Neirons[quantity_Hiden_Layers-1][count]*output_Delta[count2]
    return gradients

def sinopses_Update(speed,moment,gradients,sinopses_Change,sinopses):
    for count in range(len(sinopses_Change)): 
        sinopses_Change[count] = speed*gradients[count] + moment*sinopses_Change[count] #Расчет изменения синопса
        sinopses[count] = sinopses[count] + sinopses_Change[count] #Расчет нового синопса
    return sinopses_Change,sinopses
