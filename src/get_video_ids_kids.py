from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import re
from wonderwords import RandomWord
import random
from tqdm import tqdm


def get_vkids_ids(query=None, rango="0-4",num_random_ids=None):
    '''
    Obtiene un id de un video de youtube kids aleatorio en funcion de la query introducida y el rango de edad.
    Si no se introduce nincún rango de edad se tomara el valor por defecto 0-4.
    Si se introduce un valor para num_random_ids se devolverá una lista de ids aleatorios ignorando la query.
    
    input:
    - query: string con la query a buscar en youtube kids, si no se introduce ninguna query de obtienen los videos de la página principal de youtube kids, para el rango de edad seleccionado.
    - rango: string con el rango de edad para configurar youtube kids, puede ser "0-4", "5-8" o "9-12". Por defecto es "0-4".
    - num_random_ids: int, indica la longitud del array de ids aleatorias que devuelve
    output: 
    - video_ids: lista con los ids de los videos obtenidos
    '''
    rangos = {"0-4": 0, "5-8": 1, "9-12": 2}

    options = Options()
    # Set window size (half of 1920x1080 screen)
    options.add_argument("--window-size=960,1080")

    # Move window to right side of screen
    options.add_argument("--window-position=1400,0")

    
    driver = webdriver.Chrome(options=options)
    # #driver.minimize_window()
    driver.get(f"https://www.youtubekids.com/search")
    
    try:
        #Selección del modo padre para poder configurar youtube kids con el rango de edad deseado
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "parent-button")))
        input_element = driver.find_element(By.ID, "parent-button")
        input_element.click()
        #Next para pasar a la siguiente pantalla
        WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.ID, "next-button")))
        input_element = driver.find_element(By.ID, "next-button")
        input_element.click()
        year_digit_list = ['1','9','9','9']
        #Introducción del año de nacimiento para verificar la edad
        for i in range(4):
            WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.ID, f"onboarding-age-gate-digit-{i+1}")))
            input_element = driver.find_element(By.ID, f"onboarding-age-gate-digit-{i+1}")
            input_element.clear()
            input_element.send_keys(year_digit_list[i])
        input_element.send_keys(Keys.ENTER)
        
        #Skip del video, porque si no hay que esperar 26 segundos para poder pasar a la siguiente pantalla
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "show-text-link")))
        input_element = driver.find_element(By.ID, "show-text-link")
        input_element.click()

        #Next para pasar a la siguiente pantalla, el problema es que se van acumulando los botones next, con el mismo id, 
        #pero solo uno es visible, por lo que hay que buscar el que está visible para poder hacer click en él.
        botones = driver.find_elements(By.ID, "next-button")

        boton_visible = [b for b in botones if b.is_displayed()][0]
        boton_visible.click()

        #Skip del tutorial
        WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.ID, "skip-button")))

        botones = driver.find_elements(By.ID, "skip-button")
        boton_visible = [b for b in botones if b.is_displayed()][0]
        boton_visible.click()

        #Next para pasar a la pantalla de selección ddel rango de edad
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "next-button")))

        botones = driver.find_elements(By.ID, "next-button")
        boton_visible = [b for b in botones if b.is_displayed()][0]
        boton_visible.click()

        # En esta parte hay 3 card container que es para seleccionar entre 3 rangos de edades diferentes
        # el primero es para niños menores de 5 años, el segundo para niños entre 6 y 8 años y el tercero para niños entre 9 y 12 años. Para este caso se selecciona el segundo rango de edad.
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "card-container")))
        buttons = driver.find_elements(By.ID, "card-container")
        buttons[rangos[rango]].click()

        #Aceptar el rango de edad seleccionado
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "select-link")))
        button = driver.find_element(By.ID, "select-link")
        button.click()

        #Permitir la búsqueda de videos
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "search-on-button")))
        button = driver.find_element(By.ID, "search-on-button")
        button.click()

        #Finalizar la configuración
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "done-button")))
        button = driver.find_element(By.ID, "done-button")
        button.click()

        video_ids = set()
        lista_palabras = []
        if num_random_ids:
            num_vids = 0
            word = RandomWord()
            pbar = tqdm(total=num_random_ids)
            while len(video_ids) < num_random_ids:
                try:
                    WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "input")))
                    input_element = driver.find_element(By.ID, "input")
                    input_element.clear()
                    palabra = word.word()
                    input_element.send_keys(palabra + Keys.ENTER)
                    driver.refresh()
                    time.sleep(2.5)
                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # Buscar todos los enlaces que contienen watch?v=
                    links = soup.find_all("a", href=re.compile(r"watch\?v="))
                    link = random.choice(links)["href"].split("v=")[-1]
                    #print(link)
                    video_ids.add(link)
                    if len(video_ids)>num_vids:
                        lista_palabras.append(palabra)
                        num_vids += 1
                        pbar.update(1)
                except Exception as e: 
                    #print("Error")
                    pass
            pbar.close()

                
        else:
            #Si se desea realizar una búsqueda, se introduce la query en el buscador y se pulsa enter
            if query:
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "input")))
                input_element = driver.find_element(By.ID, "input")
                input_element.send_keys(query + Keys.ENTER)

            #Se actualiza la página ya que si no hay veces en las que no aparecen los vídeos en el html, y se espera 1 segundo a que cargue
            driver.refresh()
            time.sleep(2.5)
            # Se obtiene el código fuente de la página y se parsea con BeautifulSoup para obtener los ids de los vídeos
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Buscar todos los enlaces que contienen watch?v=
            links = soup.find_all("a", href=re.compile(r"watch\?v="))
            link = random.choice(links)["href"].split("v=")[-1]
            #print(link)
            video_ids.add(link)
        return lista_palabras, list(video_ids)
    
    finally:
        time.sleep(1)
        driver.quit()

print(get_vkids_ids(num_random_ids=20))