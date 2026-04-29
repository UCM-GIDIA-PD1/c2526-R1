''' 
Este archivo contiene la funcion para llamar al script mediante
argumentos insertados desde consola.
'''
import argparse
import extraccion.Prueba_imagenes_collect_all_data as collect_all_data_imagenes
def main():
    '''
    Llamando a este archivo desde consola se ejecuta el scripts de extracción de datos
    con el número de videos a extraer como argumento y la fecha indicada.
    '''

    # 1. Creo el parser
    parser = argparse.ArgumentParser(
        description='MENÚ DE AYUDA\n'
    )

    # 2. Agrego los argumentos
    parser.add_argument(
        "-n", "--num_videos", 
        type=int, 
        default= 500, 
        help="Número de videos a extraer"
    
    )
    parser.add_argument(
        "-f", "--fecha", 
        type=str, 
        help="Fecha de extracción"
    )

    parser.add_argument(
        "-p","--proporcion_adults", 
        type=float, default=0.5, 
        help="Proporción de videos para adultos (entre 0 y 1)"
    ) #opcional, por defecto 0.8

    parser.add_argument(
        "-i","--iteraciones", 
        type=int, default=3, 
        help="Numero de veces que se va a ejecutar el código"
    )

    # 3. Parseo los argumentos
    args = parser.parse_args()
    print("Iniciando recolección de datos con los siguientes parámetros:")
    print(f"   - Videos por iteración: {args.num_videos}")
    print(f"   - Proporción adultos:   {args.proporcion_adults}")
    print(f"   - Fecha de extracción:  {args.fecha}")
    print(f"   - Número de iteraciones:  {args.iteraciones}\n")

    # Llamo a la función de extracción de datos con los argumentos indicados
    try:
        for i in range(args.iteraciones):
            print(f'Iteracion numero {i+1} de {args.iteraciones}')
            data = collect_all_data_imagenes.collect_all_data(
                num_videos=args.num_videos,
                fecha=args.fecha,
                proporcion_adults=args.proporcion_adults
            )
    except KeyboardInterrupt:
        print("Se paró manualmente el programa")
    finally:
        print("Proceso finalizado")


if __name__ == "__main__":
    main()