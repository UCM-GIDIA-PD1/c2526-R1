''' 
Este archivo contiene la funcion para llamar al script mediante
argumentos insertados desde consola.
'''
import argparse
import collect_all_data

def main():
    '''
    Llamando a este archivo desde consola se ejecuta el scripts de extracción de datos
    con el número de videos a extraer como argumento y la fecha indicada.
    '''

    # 1. Creo el parser
    parser = argparse.ArgumentParser(
        description='MENÚ DE AYUDA\n')
    
    # MENU\n
    # -----------------------------------
    # Inserta los argumentos que quieras de la siguiente manera (si no se inserta nada tomará los valores por defecto):
    # Númerod e videos: --num_videos <número>
    # Fecha de extracción: --fecha <fecha>
    # Proporción de videos para adultos: --proporcion_adults <número entre 0 y 1>
    # -----------------------------------

    # 2. Agrego los argumentos
    parser.add_argument("archivo", help="Archivo a ejecutar", type=str)
    parser.add_argument("--num_videos", type=int, default= 500, help="Número de videos a extraer")
    parser.add_argument("--fecha", type=str, help="Fecha de extracción")
    parser.add_argument("--proporcion_adults", type=float, default=0.8, help="Proporción de videos para adultos (entre 0 y 1)") #opcional, por defecto 0.8

    # 3. Parseo los argumentos
    args = parser.parse_args()

    if args.archivo:
        print(f"Ejecutando el script {args.archivo} con los siguientes argumentos:")

    if args.num_videos:
        print(f"    Número de videos a extraer: {args.num_videos}")

    if args.fecha:
        print(f"    Fecha de extracción: {args.fecha}")

    if args.proporcion_adults:
        print(f"    Proporción de videos para adultos: {args.proporcion_adults}")
       

if (__name__ == "__main__"):
    main()