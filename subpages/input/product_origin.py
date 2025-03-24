from typing import Optional
import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import datetime

## Basic variables
# Save data to excel file
file_path = "content/plastiq_input_information.xlsx"
df_header = "Herkunftsinformationen"
wertstoff_origin_list = [ # list with the possible origins of the waste
    "Post-Industrial (PI)", "Post-Consumer (PC) – getrennte Sammlung", "Post-Consumer (PC) – gemischte Sammlung"
]
wertstoff_collection_list = [ # list with the possible ways of collection
    "regional", "lokal als Bringsystem", "haushaltsnah als Holsystem", "haushaltsgebunden als Holsystem"
]
wertstoff_collection_help = "Angabe für die Kategorie PC - getrennte Sammlung, nach welcher Sammlungsart der Wertstoff zusammengetragen wurde \n- regional (Letztbesitzer bringt Altprodukt direkt zum Erstbehandler wie z. B. bei Altfahrzeugen üblich) \n- lokal als Bringsystem (Letztbesitzer bringt z. B. Altbatterien zum Einzelhandel), \n- haushaltsnah als Holsystem (Altpapier, Glas, Textilien etc. in Containern) \n- haushaltsgebunden als Holsystem (Restabfalltonne, gelber Sack, Biotonne, blaue Tonne für Altpapier)"

## Functions
# Function to update dict with session state keys after every submission of form
def update_keys():
    for k in st.session_state.key_dict_product_origin:
        st.session_state.key_dict_product_origin[k] = st.session_state[k]

# Function to collect product origin data
def collect_product_origin(value_origin: str) -> Optional[pd.DataFrame]:
    with st.form(key="waste_origin_form"):
    
        wertstoff_verwendung = st.text_input(label="Ursprüngliche Verwendung", key="input_wertstoff_use")

        if value_origin == "Post-Consumer (PC) – getrennte Sammlung":
            wertstoff_sammlung = st.selectbox(label="Sammlungsart", options=wertstoff_collection_list, key="input_wertstoff_collection", help=wertstoff_collection_help)
            wertstoff_abfallcode = st.text_input(label="Abfallcode", max_chars=9, placeholder="XX XX XX*", key="input_wertstoff_code")
        elif value_origin == "Post-Consumer (PC) – gemischte Sammlung":
            wertstoff_sammlung = None
            wertstoff_abfallcode = st.text_input(label="Abfallcode", max_chars=9, placeholder="XX XX XX*", key="input_wertstoff_code")
        else:
            wertstoff_sammlung = None
            wertstoff_abfallcode = None

        # Additional variables for the DataFrame
        id_product = 1 #to be specified
        timestamp = datetime.datetime.now().timestamp()
        utc_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Form submit button
        submit_button_product_origin = st.form_submit_button(label='Speichern', on_click=update_keys)

        # Store the data in a DataFrame if the form is submitted
        if submit_button_product_origin:

            product_origin = {
                "ID_Wertstoff": [id_product],
                "Zeit_UTC": [utc_time],
                "Herkunftskategorie": [value_origin],
                "Ursprüngliche Verwendung": [wertstoff_verwendung],
                "Sammlungsart": [wertstoff_sammlung],
                "Abfallcode": [wertstoff_abfallcode],
            }

            product_df = pd.DataFrame(product_origin)

            # Show success message with a green checkmark
            st.success("✅ Die Angaben wurden erfolgreich übernommen!")

            return product_df
    return None

    
# For loop: Create session state key for every key in key_dict_product
for k in st.session_state.key_dict_product_origin:
    st.session_state[k] = st.session_state.key_dict_product_origin[k]

# Streamlit app
st.title("Wertstoffdaten")
st.header("Herkunft des Wertstoffs", divider="red", help="Bitte gebe die Informationen zur Herkunft, der Sammlungsart und zur ursprünglichen Verwendung des Wertstoffs an.")

wertstoff_herkunft: str = st.selectbox(label="Herkunftskategorie", options=wertstoff_origin_list, key="input_wertstoff_origin", on_change=update_keys)

# Collect contact using the function
product_df = collect_product_origin(wertstoff_herkunft)

# Display the dataframe if not None
#if product_df is not None:

    # Append the DataFrame to the existing Excel file
    #append_df_to_excel(file_path, product_df) #disabled the function to write in excel
    #show_dataframe (df_header, product_df)

# Display buttons to switch between input pages
left_column_bottom, right_column_bottom = st.columns([.13,1])
button_back = left_column_bottom.button("Zurück")
if button_back:
    st.switch_page("subpages/input/product.py")
button_next = right_column_bottom.button("Weiter")
if button_next:
    st.switch_page("subpages/input/product_quality.py")