from filter_and_divide_data import download_latest_extraction_correct, filtrar_subtitulos

df = download_latest_extraction_correct(filtrar = True)
df = filtrar_subtitulos(df)