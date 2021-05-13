import ploting
import streamlit as st
import Translations.language_names as lang_names
import Translations.translations as translates
import data.symbols_data as Stock

class GUI:
    def __init__(self):
        pass

    def Get_basic_information(self):
        lang_name = st.radio('Languages', list(lang_names.languages.values()))

        st.write(f'''
        # {translates.translate[lang_name]['main_title']}
        ### {translates.translate[lang_name]['sub_title']}
        ---
        ''')
        st.header(f'{translates.translate[lang_name]["Insert_Data"]}')
        symbol = st.selectbox(f'{translates.translate[lang_name]["Symbol"]}: ', [Stock.SYMBOLS][0])
        prediction_days = int(st.text_input(f'{translates.translate[lang_name]["Prediction_days"]}: ', 5))
        st.write(f"## {translates.translate[lang_name]['Date_Range']}")
        year = st.slider(
            f'{translates.translate[lang_name]["Select_a_range_of_Date"]}',
            2000, 2021
        )

        st.write(f'{translates.translate[lang_name]["From"]}', year, f'{translates.translate[lang_name]["until_now"]}')
        period = st.selectbox(f'{translates.translate[lang_name]["Period"]}: ', ['1d', '1w', '1m'])
        month = day = 1
        date_range = f"{year}-0{month}-0{day}"

        return symbol, prediction_days, year, period, date_range

