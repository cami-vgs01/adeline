from tkinter import filedialog
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataframe = pd.DataFrame()
def lecturaArchivo():
    global dataframe
    archivo_csv = filedialog.askopenfilename(filetypes=[("Archivo CSV", "*.csv")])
    # Leer el archivo CSV seleccionado
    try:
        dataframe.drop(index=dataframe.index, columns=dataframe.columns, inplace=True)
        datos = pd.read_csv(archivo_csv, index_col=None)
        dataframe = pd.concat([dataframe, datos], ignore_index=True)
        print("Datos ingresados correctamente")
        return dataframe
    except FileNotFoundError:
        print("No se eligió el archivo")

def calcular_suma(suma,primerasfilas, numcolumnas, numeroTeta, i,numero_aleatorio):
    E=0
    for j in range(numcolumnas):
        suma += primerasfilas.iloc[i,j] * numero_aleatorio[j]
    suma += numeroTeta
    print("Fila: ", primerasfilas.iloc[i].values)
    print("Suma: ", suma)
    #calculamos el error
    error = primerasfilas.iloc[i,-1] - suma
    E = (error)**2
    return error,E,suma

def ajustar_pesos(primerasfilas, numero_aleatorio,i, tasaprendizaje, error):
    for k in range(len(numero_aleatorio)):
        numero_aleatorio[k] += (tasaprendizaje*error*primerasfilas.iloc[i,k])
    return numero_aleatorio

def perceptron(numero_aleatorio,numeroTeta,pasadas):
    primerasfilas = dataframe.head(6)
    num_filas, num_columnas = dataframe.shape
    nombresColumnas = list(dataframe.columns)
    nombre_ultima_columna = dataframe.columns[-1]  # obtiene el nombre de la ultima columna
    palette = sns.color_palette("bright", len(primerasfilas[nombre_ultima_columna].unique()))
    color_dict = dict(zip(primerasfilas[nombre_ultima_columna].unique(), palette))
    if num_columnas-1 <3:
        dibujar2D(nombresColumnas,primerasfilas,color_dict,nombre_ultima_columna)
    elif num_columnas-1 == 3:
        dibujar3D(nombresColumnas,primerasfilas,nombre_ultima_columna)
    # Generar un número aleatorio entre 0 y 1 la librería random
    """for _ in range(num_columnas -1):
        numero_aleatorio.append(random.random())
    numeroTeta = random.random()"""
    numero_aleatorio = [0.84,0.394,0.738]
    numeroTeta = 0
    tasaAprendizaje = 0.3
    errorGlobal = []
    while pasadas > 0:
        print("Pasada: ", pasadas)
        error = []
        for i in range(primerasfilas.shape[0]):
            suma = 0
            errorInd,E,suma = calcular_suma(suma,primerasfilas, num_columnas-1, numeroTeta,i,numero_aleatorio)
            print("Error: ", errorInd)
            if errorInd != primerasfilas.iloc[i,-1]:
                suma = 0
                numero_aleatorio = ajustar_pesos(primerasfilas, numero_aleatorio,i,tasaAprendizaje,errorInd)
                print("Pesos:", numero_aleatorio)
                # Verificar si la fila actual coincide después del ajuste
                suma_despues_ajuste,E,suma = calcular_suma(suma,primerasfilas, num_columnas-1, numeroTeta,i,numero_aleatorio)
                print("Y despues del ajuste: ", suma_despues_ajuste)

                error.append(E)
            else:
                error.append(E)
        print(error)
        errorGlobal.append(1/2*sum(error))

        pasadas = pasadas - 1

    print("Error total: ", errorGlobal)
    #Dibujar el error
    plt.plot(errorGlobal)
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.show()
    print("Pesos finales: ", numero_aleatorio)
    print("Teta final: ", numeroTeta)
    return numero_aleatorio, numeroTeta



def predecir(numero_aleatorio, numeroTeta, num):
    if num == 1:
        print("Ingrese los datos para predecir:")
        num_columnas = dataframe.shape[1]
        datos = []
        for i in range(num_columnas - 1):
            valor = float(input(f"Ingrese el valor para la columna {i+1}: "))
            datos.append(valor)
        resto_filas = pd.DataFrame([datos], columns=dataframe.columns[:-1])
    else:
        resto_filas = dataframe.iloc[6:]

    num_filas, num_columnas = dataframe.shape
    predicciones = []  # Lista para almacenar los resultados de las predicciones
    for i in range(resto_filas.shape[0]):
        suma = 0
        y, E,suma = calcular_suma(suma,resto_filas, num_columnas-1, numeroTeta,i,numero_aleatorio)
        # Agregar el resultado de la predicción a la lista de predicciones
        predicciones.append(suma)
    dfModificado = resto_filas.copy()
    # Agregar la lista de predicciones como una nueva columna al DataFrame
    dfModificado['Prediccion'] = predicciones
    # Imprimir el DataFrame con las predicciones
    print(dfModificado)

def dibujar2D(nombres_columnas,sub_df,color_dict,nombre_ultima_columna):
    sns.scatterplot(x=nombres_columnas[0], y=nombres_columnas[1], hue=nombre_ultima_columna, data=sub_df, palette=color_dict)
    plt.scatter(sub_df.iloc[:,0], sub_df.iloc[:,1], c=sub_df[nombre_ultima_columna].apply(lambda x: color_dict[x]), marker='s')
    plt.xlabel(nombres_columnas[0])
    plt.ylabel(nombres_columnas[1])
    plt.show()

def dibujar3D(nombres_columnas, df, nombre_ultima_columna):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    clases = df[nombre_ultima_columna].unique()
    color_dict = {clase: np.random.rand(3,) for clase in clases} # generamos un diccionario con un color aleatorio por cada clase
    for clase, color in color_dict.items():
        temp_df = df[df[nombre_ultima_columna] == clase]
        ax.scatter(temp_df[nombres_columnas[0]], temp_df[nombres_columnas[1]], temp_df[nombres_columnas[2]], color=color, marker='s', label=str(clase))
    ax.set_xlabel(nombres_columnas[0])
    ax.set_ylabel(nombres_columnas[1])
    ax.set_zlabel(nombres_columnas[2])
    plt.legend()
    plt.show()

def menu():
    numero_aleatorio = []
    numeroTeta = 0
    while True:
        print("1. Cargar archivo")
        print("2. Visualizar datos")
        print("3. Entrenar modelo")
        print("4. Predecir")
        print("5. Salir")
        try:
            opcion = int(input("Ingrese la opcion: "))
            if opcion == 1:
                try:
                    dataframe.drop(index=dataframe.index, columns=dataframe.columns, inplace=True)
                    # Crear la ventana principal
                    root = tk.Tk()
                    # Agregar un botón para abrir el cuadro de diálogo de selección de archivos
                    boton_abrir = tk.Button(root, text="Abrir archivo CSV", command=lecturaArchivo)
                    boton_abrir.pack()
                    # Mostrar la ventana principal
                    root.mainloop()
                except:
                    print("No se eligió el archivo")
            elif opcion == 2:
                print(dataframe)
            elif opcion == 3:
                print("Ingrese en número de pasadas del modelo")
                try:
                    pasadas = int(input())
                    numero_aleatorio, numeroTeta=perceptron(numero_aleatorio,numeroTeta, pasadas)
                except ValueError:
                    print("Ingrese un valor númerico")
            elif opcion == 4:
                print("Seleccione la fuente de datos:")
                print("1. Resto de filas")
                print("2. Ingresar desde consola")
                try:
                    seleccion_fuente = int(input("Ingrese la opción: "))
                    if seleccion_fuente == 2:
                        predecir(numero_aleatorio, numeroTeta,1)
                    else:
                        predecir(numero_aleatorio, numeroTeta,2)
                except ValueError:
                    print("Opción inválida")
            elif opcion == 5:
                break
        except ValueError:
            print("Opcion invalida")


menu()