## Import libraries
from typing import List, Literal, Never, Optional, Tuple
import streamlit as st
import pandas as pd
import math

def bad_input_error(msg: str, page: None | str = None) -> Never:
    st.error(msg)
    if page is not None:
        if st.button('Zu Eingabe'):
            st.switch_page(page)
    st.stop()

def bad_input_warning(msg: str, page: None | str = None):
    st.warning(msg)
    if page is not None:
        if st.button('Zu Eingabe', key=msg):
            st.switch_page(page)

## Define variables
# Initialize variables
material_score = 0 # variable for the final score of the waste material
percent_valuable_substance = 0.0 #set up variable for the amount of valuable substance in the waste
material_type: List[str] = [] # list of waste types added during the input
material_share: List[float] = [] # list of corresponding waste percentages added during the input
comparability_malus = 0 # initiate comparability malus

# file paths
file_path_background = "content/background_data_decision_tree.xlsx"
file_path_input = "content/plastiq_input_information.xlsx"
file_path_material_mapping = "content/material_mapping.csv"
file_path_material_worth = "content/worth_materials.csv"

# this table maps the kinds of material in companies.csv
# to those from the decision tree; materials which are
# not present in the mapping are only present in one of the
# two tables and thus cannot be mapped.
mat_map: pd.DataFrame = pd.read_csv(file_path_material_mapping, sep=';', header=None)
mat_map.columns = ['companies.csv', 'decision_tree']
mat_map: pd.DataFrame = mat_map.apply(lambda x: list(x.str.split(','))) # type: ignore

companies = pd.read_csv("content/companies.csv")
companies_mats = set(map(lambda x: x.strip(), sum(companies['materials'].apply(lambda x: x.split(',')), start=[])))

def map_material(mat: str,
        src: Literal['decision_tree', 'companies.csv'],
        dst: Literal['decision_tree', 'companies.csv'])\
        -> Optional[List[str]]:
    assert src != dst, 'src and dst must be different (tried to map from {} to {})'.format(src, dst)
    found = mat_map[src].apply(lambda x: mat in x)
    row: pd.DataFrame = mat_map[found] # type: ignore
    if row.empty:
        return None
    assert len(row) == 1, f'More than one material matched for{mat} ({src}->{dst})'
    return row[dst].iloc[0]

import math
mat_worth_df = pd.read_csv(file_path_material_worth, sep=';')
mat_worth: dict[str, int] = {}
for (mat, eur) in zip(mat_worth_df['material_mapped'], mat_worth_df['worth_eur']):
    if isinstance(mat, float): # is nan / empty field
        continue
    for m in mat.split(','):
        if m:
            mat_worth[m] = eur

# get parameters for the sorting and lca algorithm
# function to call parameters to dict
def load_parameters_from_excel_to_dict(file_path: str, sheet_name: str) -> dict:
    """
    Loads parameters from an Excel sheet into a dictionary.

    Parameters:
        file_path (str): Path to the Excel file containing the parameters.
        sheet_name (str): Sheet name for the relevant Excel data

    Returns:
        dict: A dictionary with keys as parameter names and values as their corresponding values.
    """
    # Load Excel into a DataFrame
    df = pd.read_excel(file_path, sheet_name)

    # Convert DataFrame to dictionary
    parameters_dict = pd.Series(df["value"].values, index=df["parameter"]).to_dict()

    return parameters_dict

# function to set sort parameters
def set_sort_parameters_from_dict(parameters: dict):
    """
    Extracts sort parameters from a dictionary and sets them as variables.

    Parameters:
        parameters (dict): Dictionary containing parameter keys and values.
    """
    # Example of dynamically setting parameters
    global set_threshold_assessibility, wt_ferromagnetic, wt_eddycurrent, wt_density
    global wt_electrostatic, wt_sensorbased, ws_boundary_min, ws_boundary_g_f, ws_boundary_f_e    
    global ws_boundary_e_d, ws_boundary_d_c, ws_boundary_c_b, ws_boundary_b_a, ws_boundary_max, category_boundaries
    global comparability_malus_min, comparability_malus_max, contamination_malus_min, contamination_malus_max
    
    set_threshold_assessibility = parameters.get("set_threshold_assessibility")
    wt_ferromagnetic = parameters.get("wt_ferromagnetic")
    wt_eddycurrent = parameters.get("wt_eddycurrent")
    wt_density = parameters.get("wt_density")
    wt_electrostatic = parameters.get("wt_electrostatic")
    wt_sensorbased = parameters.get("wt_sensorbased")
    
    ws_boundary_min = parameters.get("ws_boundary_min")
    ws_boundary_g_f = parameters.get("ws_boundary_g_f")
    ws_boundary_f_e = parameters.get("ws_boundary_f_e")
    ws_boundary_e_d = parameters.get("ws_boundary_e_d")
    ws_boundary_d_c = parameters.get("ws_boundary_d_c")
    ws_boundary_c_b = parameters.get("ws_boundary_c_b")
    ws_boundary_b_a = parameters.get("ws_boundary_b_a")
    ws_boundary_max = parameters.get("ws_boundary_max")

    category_boundaries = [
        ws_boundary_min, ws_boundary_g_f, ws_boundary_f_e, ws_boundary_e_d,
        ws_boundary_d_c, ws_boundary_c_b, ws_boundary_b_a, ws_boundary_max
    ]
    
    comparability_malus_min = parameters.get("comparability_malus_min")
    comparability_malus_max = parameters.get("comparability_malus_max")
    contamination_malus_min = parameters.get("contamination_malus_min")
    contamination_malus_max = parameters.get("contamination_malus_max")

# set sort parameters
sort_parameters_dict = load_parameters_from_excel_to_dict(file_path=file_path_background, sheet_name="sort_parameters")
set_sort_parameters_from_dict(parameters=sort_parameters_dict)

# function to set lca parameters
def set_lca_parameters_from_dict(parameters: dict):
    """
    Extracts LCA parameters from a dictionary and sets them as variables.

    Parameters:
        parameters (dict): Dictionary containing parameter keys and values.
    """
    # Example of dynamically setting parameters
    global lca_declared_unit, lca_emission_origen
    global lca_substitution_factor_plastic, lca_substitution_factor_metal
    global lca_substitution_factor_electricity, lca_substitution_factor_heat
    global lca_distance_to_wte, lca_distance_to_landfill, lca_distance_to_recycling_metal
    global lca_vehicle_to_wte, lca_vehicle_to_landfill, lca_vehicle_to_recycler_metal, lca_vehicle_to_recycler_plastic
    global lca_efficiency_wte_electric, lca_efficiency_wte_heat, lca_share_dross
    global lca_electricity_use_sorting_metal, lca_electricity_use_shredding_metal, lca_electricity_use_cleaning_metal
    global lca_electricity_use_melting_metal, lca_heat_use_cleaning_metal, lca_water_use_cleaning_metal
    global lca_wastewater_use_cleaning_metal, lca_solvent_use_cleaning_metal
    global lca_electricity_use_sorting_mixed, lca_electricity_use_shredding_plastic, lca_electricity_use_cleaning_plastic
    global lca_heat_use_cleaning_plastic, lca_water_use_cleaning_plastic, lca_wastewater_use_cleaning_plastic
    global lca_solvent_use_cleaning_plastic, lca_share_io_shredding, lca_share_io_cleaning, lca_share_io_regranulation

    # Assigning variables from dictionary
    lca_declared_unit = parameters.get("lca_declared_unit")
    lca_emission_origen = parameters.get("lca_emission_origen")
    lca_substitution_factor_plastic = parameters.get("lca_substitution_factor_plastic")
    lca_substitution_factor_metal = parameters.get("lca_substitution_factor_metal")
    lca_substitution_factor_electricity = parameters.get("lca_substitution_factor_electricity")
    lca_substitution_factor_heat = parameters.get("lca_substitution_factor_heat")
    lca_distance_to_wte = parameters.get("lca_distance_to_wte")
    lca_distance_to_landfill = parameters.get("lca_distance_to_landfill")
    lca_distance_to_recycling_metal = parameters.get("lca_distance_to_recycling_metal")
    lca_vehicle_to_wte = parameters.get("lca_vehicle_to_wte")
    lca_vehicle_to_landfill = parameters.get("lca_vehicle_to_landfill")
    lca_vehicle_to_recycler_metal = parameters.get("lca_vehicle_to_recycler_metal")
    lca_vehicle_to_recycler_plastic = parameters.get("lca_vehicle_to_recycler_plastic")
    lca_efficiency_wte_electric = parameters.get("lca_efficiency_wte_electric")
    lca_efficiency_wte_heat = parameters.get("lca_efficiency_wte_heat")
    lca_share_dross = parameters.get("lca_share_dross")
    lca_electricity_use_sorting_metal = parameters.get("lca_electricity_use_sorting_metal")
    lca_electricity_use_shredding_metal = parameters.get("lca_electricity_use_shredding_metal")
    lca_electricity_use_cleaning_metal = parameters.get("lca_electricity_use_cleaning_metal")
    lca_electricity_use_melting_metal = parameters.get("lca_electricity_use_melting_metal")
    lca_heat_use_cleaning_metal = parameters.get("lca_heat_use_cleaning_metal")
    lca_water_use_cleaning_metal = parameters.get("lca_water_use_cleaning_metal")
    lca_wastewater_use_cleaning_metal = parameters.get("lca_wastewater_use_cleaning_metal")
    lca_solvent_use_cleaning_metal = parameters.get("lca_solvent_use_cleaning_metal")
    lca_electricity_use_sorting_mixed = parameters.get("lca_electricity_use_sorting_mixed")
    lca_electricity_use_shredding_plastic = parameters.get("lca_electricity_use_shredding_plastic")
    lca_electricity_use_cleaning_plastic = parameters.get("lca_electricity_use_cleaning_plastic")
    lca_heat_use_cleaning_plastic = parameters.get("lca_heat_use_cleaning_plastic")
    lca_water_use_cleaning_plastic = parameters.get("lca_water_use_cleaning_plastic")
    lca_wastewater_use_cleaning_plastic = parameters.get("lca_wastewater_use_cleaning_plastic")
    lca_solvent_use_cleaning_plastic = parameters.get("lca_solvent_use_cleaning_plastic")
    lca_share_io_shredding = parameters.get("lca_share_io_shredding")
    lca_share_io_cleaning = parameters.get("lca_share_io_cleaning")
    lca_share_io_regranulation = parameters.get("lca_share_io_regranulation")

# set lca parameters
lca_parameters_dict = load_parameters_from_excel_to_dict(file_path=file_path_background, sheet_name="lca_parameters")
set_lca_parameters_from_dict(parameters=lca_parameters_dict)

# calculated lca parameters
lca_share_io_overall = lca_share_io_shredding * lca_share_io_cleaning * lca_share_io_regranulation # share of i.O. material fit for recycling overall

# Define labels for different dataframes
# for dataframe materials
label_material_type = "type"
label_material_category = "category"
label_material_share = "share"
label_material_result_sorting = "result_sorting"
label_material_result_compatibility = "result_compatibility"
columns_materials = [label_material_type, label_material_category, label_material_share, label_material_result_sorting, label_material_result_compatibility]

# for dataframe of sorting results:
label_material_pairing = "material_pairing"
label_sort_ferromagnetic = "sort_ferromagnetic"
label_sort_eddycurrent = "sort_eddycurrent"
label_sort_density = "sort_density"
label_sort_electrostatic = "sort_electrostatic"
label_sort_sensorbased = "sort_sensorbased"
label_sort_compatibility = "sort_compatibility"
columns_sorting = [label_material_pairing, label_sort_ferromagnetic, label_sort_eddycurrent, label_sort_density, label_sort_electrostatic, label_sort_sensorbased, label_sort_compatibility]

# for dataframe emissions
columns_emissions = [
    "emissions_overall_per_year", "emissions_overall_per_weight", "emissions_overall_per_declared_unit",
    "processing_plastic_total", "processing_plastic_sorting", "processing_plastic_shredding", "processing_plastic_cleaning", "processing_plastic_drying", "processing_plastic_regranulation", 
    "processing_metal_total", "processing_metal_sorting", "processing_metal_shredding", "processing_metal_cleaning", "processing_metal_melting",
    "endoflife_total", "endoflife_incineration", "endoflife_landfill",
    "transport_total", "transport_recycler_plastic", "transport_recycler_metal", "transport_wte", "transport_landfill",
    "avoided_total", "avoided_electricity", "avoided_heat", "avoided_plastic", "avoided_metal"
] 

## Get session state variables
# Get weight of materials
waste_weigth = st.session_state.key_dict_product_amount["input_wertstoff_menge"]
# Get weight of material per year
material_weight_regular = st.session_state.key_dict_product_amount["input_haeufigkeit_menge"]
material_frequency = st.session_state.key_dict_product_amount["input_haeufigkeit_turnus"]
# Get reach conformity
reach_conformity = st.session_state.key_dict_product_quality["input_wertstoff_reach"]
# Get number of fractions
material_fraction_number = st.session_state.key_dict_product["input_waste_fraction_number"]
# Get material type and share of fraction to total material
for k in range(material_fraction_number):
    material_type.append(st.session_state.key_dict_product[f"input_wertstoff_typ_{k}"])
    material_share.append(st.session_state.key_dict_product[f"input_wertstoff_anteil_{k}"])

if not math.isclose(sum(material_share), 100.0):
    bad_input_error("Die Summe der Wertstoffanteile muss 100% ergeben, bitte korrigiere die Anteile", "subpages/input/product.py")

# get material contamination
material_contamination = st.session_state.key_dict_product_quality["input_wertstoff_contaminants_level"]

## Get input data from files
# Get list with all available materials and data on them
df_materials = pd.read_excel(file_path_background, sheet_name = "list_material")
# Magnetabscheidung
df_sort_ferromagnetic = pd.read_excel(file_path_background, sheet_name = "sort_ferromagnetic", index_col=0)
# Wirbelstromsortierung
df_sort_eddycurrent = pd.read_excel(file_path_background, sheet_name = "sort_eddycurrent", index_col=0)
# Dichtesortierung
df_sort_density = pd.read_excel(file_path_background, sheet_name = "sort_density", index_col=0)
# Elektrostatische Sortierung
df_sort_electrostatic = pd.read_excel(file_path_background, sheet_name = "sort_electrostatic", index_col=0)
# Sensorbased Sorting 
df_sort_sensorbased = pd.read_excel(file_path_background, sheet_name = "sort_sensorbased", index_col=0)
# Material compatibility
df_sort_compatibility = pd.read_excel(file_path_background, sheet_name = "sort_compatibility", index_col=0)
# Get emission data
df_lca_emission_data = pd.read_excel(file_path_background, sheet_name = "lca_calculation")

## Define functions - general
def func_initializeOutputDf(amount: int, label_columns: list[str], material_type: list, label_material_pairing: str = "material_pairing") -> pd.DataFrame:
    """
    Initializes a DataFrame for sorting output scores by creating all unique material pairings.

    Parameters:
        amount (int): The number of materials to compare.
        label_columns (list): List of column labels for the new DataFrame.
        material_type (list): List of material types (e.g., ["PE", "PP", "ABS"]).
        label_material_pairing (str): The column name for storing material pairings.

    Returns:
        pd.DataFrame: A DataFrame initialized with unique material pairings in the specified column.

    Raises:
        ValueError: If the input lists have mismatched lengths or if required parameters are missing.
    """
    # Input validation
    if amount != len(material_type):
        raise ValueError("The 'amount' parameter must match the length of 'material_type'.")
    if not label_columns:
        raise ValueError("The 'label_columns' parameter must not be empty.")

    # Initialize DataFrame
    df_new = pd.DataFrame(columns=label_columns)
    list_material_pairing = []

    # Generate unique material pairings
    for k in range(amount):
        for l in range(k + 1, amount):
            # Specify the row and column indices
            first_material = material_type[k]
            second_material = material_type[l]

            # Define material pairing
            pair_string = f"{first_material}_{second_material}"

            # Add to list
            list_material_pairing.append(pair_string)

    # Add material pairings to the DataFrame
    df_new[label_material_pairing] = list_material_pairing

    # Output new DataFrame
    return df_new

# Initialize dataframe for result 
def func_initializeResultDf(amount: int = material_fraction_number, label_columns: list[str] = columns_materials) -> pd.DataFrame:
    # New dataframe with material as column name
    df_new = pd.DataFrame(columns = label_columns)
    # Add first column "material type"
    df_new[label_material_type] = material_type
    # Lookup category of material
    material_category = []

    for k in range(amount):
        if material_type[k] is None:
            bad_input_error(f"Materialtyp #{k + 1} ist nicht eingegeben, überprüfe deine Eingabe", "subpages/input/product.py")
        # lookup category from material dataframe
        pat = df_materials["abbreviation"].str.fullmatch(material_type[k], case=False, na=False)
        category = df_materials.loc[pat, "category"]
        material_category.append(category.iloc[0])

    # Add second column with material category
    df_new[label_material_category] = material_category

    # Add third column with material share
    df_new[label_material_share] = material_share            

    return df_new

def func_get_indices_for_material(df: pd.DataFrame, column: str, material: str) -> list:
    """
    Get the indices of rows where a specific material is present in a specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame to search.
        column (str): The column name to search in.
        material (str): The material to look for.

    Returns:
        list: A list of row indices where the material is present.
    """
    indices = df[df[column].str.contains(material, case=False, na=False)].index
    return indices.tolist()

# Function to check all available materials for their sorting options
def func_checkSorting(df: pd.DataFrame, amount: float = material_fraction_number) -> list:
    
    # Initialize list with all result values
    values_sort = []

    # Loop through each entry and compare it with every other entry, while avoiding duplicates
    for k in range(amount):
        for l in range(k+1, amount):
              
          # Specify the row and column indices
          row_label = material_type[k]
          column_label = material_type[l]

          # Get and return value at indices from matrix
          result = df.loc[row_label, column_label]

          # Add to list
          values_sort.append(float(result))
    
    # Output list with result values
    return values_sort

def func_sum_columns_with_conditions(df, row_label, include_substring, exclude_substring) -> float:
    """
    Sums values in a row for columns containing a specific substring
    but excluding columns containing another substring.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        row_label (str): The label of the row to sum.
        include_substring (str): Substring to include in column names.
        exclude_substring (str): Substring to exclude from column names.

    Returns:
        float: The sum of the selected values.
    """
    filtered_columns = [
        col for col in df.columns
        if include_substring in col and exclude_substring not in col
    ]
    return df.loc[row_label, filtered_columns].sum()

# Calculate the weighted mean of the results from sorting
def func_calculate_weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """
    Calculates the weighted mean of a column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        value_col (str): The name of the column with the values to average.
        weight_col (str): The name of the column with the weights.

    Returns:
        float: The weighted mean.
    """
    weighted_mean = (
        (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()
    )
    return weighted_mean

# Map the weighted mean to a scale
def func_map_to_scale(value: float, return_if_1: float, start_scale: float, end_scale: float) -> float:
    """
    Maps a given value to a corresponding scale with configurable parameters.

    Parameters:
        value (float): The input value to be mapped.
        return_if_1 (float): The scale value to return if the input value is exactly 1.
        start_scale (float): The starting value of the scale for input values < 1.
        end_scale (float): The ending value of the scale for input values < 1.

    Returns:
        float: The value mapped to the scale.

    Raises:
        ValueError: If the input value is greater than 1.
    """
    if value == 1:
        # If the value is exactly 1, return the specified return_if_1 value
        return return_if_1
    elif value < 1:
        # Map the value evenly to the specified scale for values < 1
        scaled_value = start_scale + (value / 1) * (end_scale - start_scale)
        return scaled_value
    else:
        raise ValueError("The input value must be less than or equal to 1.")

# calculate mean of certain values for a list of materials
def func_calculate_mean_for_materials(df: pd.DataFrame, materials: list, material_col: str, value_col: str) -> list:
    """
    Calculates the mean of values in a column for rows containing a specific material in the another column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        materials (list): A list of material names to search for in the column.
        material_col (str): The name of the column containing material types.
        value_col (str): The name of the column containing the values to average.

    Returns:
        list: A list of mean values corresponding to each material in the input list.
    """
    mean_values = []

    for material in materials:
        # Filter rows containing the material in the specified column
        filtered_rows = df[df[material_col].str.contains(material, case=False, na=False)]

        # Calculate the mean of the specified column for the filtered rows
        mean_value = filtered_rows[value_col].mean()

        # Append the mean value to the result list
        mean_values.append(mean_value)

    return mean_values

# calculate cases for contamination malus
def func_calculate_contamination_malus(material_contamination: str, scale_contamination_malus_min: float, scale_contamination_malus_max: float) -> float:
    """
    Calculates the malus value for a material contamination level based on a specified scale.

    Parameters:
        material_contamination (str): The level of contamination.
                                       Expected values: "keine", "gering", "mittel", "hoch", "sehr hoch".
        scale_contamination_malus_min (float): The minimum malus value (for "keine").
        scale_contamination_malus_max (float): The maximum malus value (for "sehr hoch").

    Returns:
        float: The malus value corresponding to the contamination level.

    Raises:
        ValueError: If the input does not match any expected contamination levels.
    """
    # Define contamination levels in order of severity
    contamination_levels = ["keine", "gering", "mittel", "hoch", "sehr hoch"]

    # Check if the input is valid
    if material_contamination not in contamination_levels:
        bad_input_error("Der Kontaminationswert (Verschmutzungsgrad) fehlt, überprüfe deine Eingabe.", page="subpages/input/product_quality.py")

    # Calculate the step size
    num_levels = len(contamination_levels)
    step_size = (scale_contamination_malus_max - scale_contamination_malus_min) / (num_levels - 1)

    # Find the index of the contamination level
    level_index = contamination_levels.index(material_contamination)

    # Calculate the malus value
    malus_value = scale_contamination_malus_min + level_index * step_size

    return malus_value

# Function to locate WS to a class
def func_evaluateWS(material_score: float, category_boundaries: list[float] = category_boundaries) -> tuple[str, str]:
    """
    Categorizes a material score into a waste classification based on provided boundaries
    and returns the classification and description.

    Parameters:
        material_score (float): The score of the material to be evaluated.
        category_boundaries (list): A list containing category boundaries:
            - [min_score, g_f, f_e, e_d, d_c, c_b, b_a, max_score]

    Returns:
        tuple[str, str]: A tuple containing:
            - ws_category: The classification of the material (A to H).
            - ws_sentence: A descriptive sentence explaining the classification.

    Raises:
        ValueError: If the input score is outside the expected range or the list length is invalid.
    """
    # Validate the category boundaries
    if len(category_boundaries) != 8:
        raise ValueError("The category_boundaries list must contain exactly 8 values.")
    
    # Unpack category boundaries
    min_score, g_f, f_e, e_d, d_c, c_b, b_a, max_score = category_boundaries

    # Validate input score
    if not (min_score <= material_score <= max_score):
        raise ValueError(f"Material score must be between {min_score} and {max_score}.")

    # Round the score to one decimal place
    ws_rounded = round(material_score, 1)

    # Classify the material based on its score
    if min_score <= ws_rounded < g_f:  # Klasse H: Sondermüll
        ws_category = "H"
        ws_sentence = "Dein Abfall sollte aufgrund der Inhalte auf dem Sondermüll entsorgt werden."
    elif g_f <= ws_rounded < f_e:  # Klasse F: Energetische Verwertung
        ws_category = "F"
        ws_sentence = "Dein Abfall eignet sich voraussichtlich nur für die energetische Verwertung."
    elif f_e <= ws_rounded < e_d:  # Klasse E: Recycling, chemisch
        ws_category = "E"
        ws_sentence = "Deine Wertstoffe eignen sich voraussichtlich für das chemische Recycling."
    elif e_d <= ws_rounded < d_c:  # Klasse D: Recycling, werkstofflich, gemischt
        ws_category = "D"
        ws_sentence = "Deine Wertstoffe eignen sich voraussichtlich für das gemischte mechanische Recycling."
    elif d_c <= ws_rounded < c_b:  # Klasse C: Recycling, werkstofflich, sortenrein
        ws_category = "C"
        ws_sentence = "Deine Wertstoffe eignen sich voraussichtlich für das sortenreine mechanische Recycling."
    elif c_b <= ws_rounded < b_a:  # Klasse B: Wiederverwendung, niederwertig
        ws_category = "B"
        ws_sentence = "Deine Wertstoffe können mit einer anderen Funktion wiederverwertet werden."
    elif b_a <= ws_rounded <= max_score:  # Klasse A: Wiederverwendung, gleichwertig
        ws_category = "A"
        ws_sentence = "Deine Wertstoffe können mit der gleichen Funktion wiederverwertet werden."
    else:  # Should not occur due to validation but left for safety
        ws_category = "Eingabe konnte nicht verarbeitet werden."
        ws_sentence = "Die Eingabe liegt außerhalb des erwarteten Bereichs."

    return ws_category, ws_sentence

## Define functions - LCA
# Get a specific emission factor from background_data
def func_lca_get_emission_factor(label_category: str, label_use: str) -> float:
    """
    Get a specific emission factor from the background data based on category and use label.

    Parameters:
        label_category (str): The category to filter the emission data by.
        label_use (str): The use label to search for in the filtered emission data.

    Returns:
        float: The emission factor (GWP100) corresponding to the specified category and use label.

    Raises:
        ValueError: If no emission factor is found for the given category and use label.
    """
    # Filter the DataFrame by the specified category
    df_filtered = df_lca_emission_data[df_lca_emission_data["category"] == label_category]

    # Lookup emission factor specific to the label_use
    emission_factor = df_filtered.loc[
        df_filtered["use for"].str.contains(label_use, case=False, na=False), "GWP100"
    ]

    # Check if an emission factor was found
    if emission_factor.empty:
        raise ValueError(
            f"No emission factor found for category '{label_category}' and use label '{label_use}'."
        )

    # Return the first matching emission factor (or handle multiple matches if required)
    return float(emission_factor.iloc[0])

# get combined share of a specific category (plastic, meta)
def func_lca_get_category_share(df: pd.DataFrame, relevant_category: str) -> tuple:
    """
    Calculates the combined share of a specific category in a DataFrame and returns
    both the filtered DataFrame and the combined share.

    Parameters:
        df (pd.DataFrame): The DataFrame containing material data.
        relevant_category (str): The category for which the share is calculated (e.g., 'plastic', 'metal').

    Returns:
        tuple: A tuple containing:
            - float: The total combined share of the specified category <= 1.0
            - pd.DataFrame: The filtered DataFrame containing only rows of the relevant category.

    Raises:
        ValueError: If the DataFrame does not contain the required columns or the category is not found.
    """
    # Ensure required columns exist in the DataFrame
    required_columns = [label_material_category, label_material_share]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    # Filter DataFrame for the relevant category
    df_filtered = df[df[label_material_category] == relevant_category]

    # Sum of the share of relevant materials
    category_share = df_filtered[label_material_share].sum()

    # Return the combined share and the filtered DataFrame
    return float(category_share/100), df_filtered

# Get emissions from transport
def func_lca_emissions_transport(label_use: str, distance: float = 100.0, payload: float = 1.0) -> float:
    """
    Calculates the emissions from transport based on the distance, payload, and emission factor.

    Parameters:
        label_use (str): The specific transport mode or label (e.g., "truck", "train", "ship").
        distance (float): The distance traveled in kilometers. Default is 100.0 km.
        payload (float): The payload weight in tons. Default is 1.0 ton.

    Returns:
        float: The calculated emissions in t CO2e.

    Raises:
        ValueError: If the distance or payload is negative, as these are invalid inputs.
    """
    # Validate inputs
    if distance < 0:
        raise ValueError("Distance must be a non-negative value.")
    if payload <= 0:
        raise ValueError("Payload must be greater than zero.")

    # Get emission factor from the external function
    emission_factor = func_lca_get_emission_factor(label_category="transport", label_use=label_use)

    # Validate the emission factor
    if emission_factor is None or emission_factor < 0:
        raise ValueError(f"Invalid emission factor received for label_use '{label_use}'.")

    # Calculate emissions: payload (t) * distance (km) * emission factor (kg CO2e / (t * km))
    emission_transport = payload * distance * emission_factor / 1000

    return emission_transport

# get emissions from waste incineration
def func_lca_emissions_incineration(df: pd.DataFrame, weight: float) -> float:
    """
    Calculates the total emissions from waste incineration based on material fractions and their respective emission factors.

    Parameters:
        df (pd.DataFrame): The DataFrame containing material types and their shares.
                          It must have columns for material type and material share.
        weight (float): The total weight of the waste in kilograms.

    Returns:
        float: Total emissions from waste incineration in kg CO2e.

    Raises:
        ValueError: If required columns are missing in the DataFrame.
    """
    # Validate required columns
    required_columns = [label_material_type, label_material_share]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must contain the column '{col}'.")

    # Initialize total emissions
    total_emissions = 0.0  # in kg CO2e

    # Iterate through each material fraction
    for k in range(len(df)):
        # Extract material type and share
        material_type = df.at[k, label_material_type]
        material_share = float(df.at[k, label_material_share]) / 100

        # Lookup emission factor for the material type
        emission_factor = func_lca_get_emission_factor(label_category="incineration", label_use=material_type)

        # Update total emissions
        total_emissions += weight * material_share * emission_factor

    return total_emissions

# get energy from waste incineration
def func_lca_energy_incineration(
    df: pd.DataFrame,
    weight: float,
    df_materials: pd.DataFrame = df_materials,
    lca_efficiency_wte_electric: float = lca_efficiency_wte_electric,
    lca_efficiency_wte_heat: float = lca_efficiency_wte_heat
) -> tuple[float, float]:
    """
    Calculates the electric and heat energy generated from waste incineration.

    Parameters:
        df (pd.DataFrame): DataFrame containing material types and their shares.
                           Must have columns "label_material_type" and "label_material_share".
        weight (float): Total weight of waste in kilograms.
        df_materials (pd.DataFrame): DataFrame containing material properties, including heating values.
                                     Must have columns "abbreviation" and "lower_heating_value_MJ_per_kg".
        lca_efficiency_wte_electric (float): Efficiency factor for electricity generation from waste incineration.
        lca_efficiency_wte_heat (float): Efficiency factor for heat generation from waste incineration.

    Returns:
        tuple[float, float]: A tuple containing:
            - Electric energy generated in kWh.
            - Heat energy generated in kWh.

    Raises:
        ValueError: If required columns are missing in either DataFrame.
    """
    # Validate required columns in df
    required_columns_df = [label_material_type, label_material_share]
    for col in required_columns_df:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must contain the column '{col}'.")

    # Validate required columns in df_materials
    required_columns_materials = ["abbreviation", "lower_heating_value_MJ_per_kg"]
    for col in required_columns_materials:
        if col not in df_materials.columns:
            raise ValueError(f"The df_materials DataFrame must contain the column '{col}'.")

    # Initialize total electric and heat energy
    electric_energy_incineration = 0.0  # in kWh
    heat_energy_incineration = 0.0  # in kWh

    # Iterate through each material fraction
    for k in range(len(df)):
        # Extract material type and share
        material_type = df.at[k, label_material_type]
        material_share = float(df.at[k, label_material_share]) / 100

        # Lookup lower heating value (MJ/kg) for the material
        matching_row = df_materials.loc[df_materials["abbreviation"].str.fullmatch(material_type, case=False, na=False)]
        if matching_row.empty:
            raise ValueError(f"No matching heating value found for material type '{material_type}'.")
        lower_heating_value = float(matching_row["lower_heating_value_MJ_per_kg"])

        # Calculate electric and heat energy (converted from MJ to kWh using 1 kWh = 3.6 MJ)
        electric_energy_incineration += (
            lower_heating_value * weight * material_share * lca_efficiency_wte_electric / 3.6
        )
        heat_energy_incineration += (
            lower_heating_value * weight * material_share * lca_efficiency_wte_heat / 3.6
        )

    return electric_energy_incineration, heat_energy_incineration

# get emissions from process (categories: electricity, heat, ressources)
def func_lca_emissions_process(category: str, origen: str, source: str, amount: float = 1.0) -> float:
    """
    Calculates emissions from a process based on the category, origin, source, 
    and the amount of the process.

    Parameters:
        category (str): The category of the process (e.g., 'electricity', 'heat', 'resources').
        origen (str): The origin of the emission factor (e.g., geographical location).
        source (str): The specific source of emissions (e.g., type of fuel or material).
        amount (float): The process amount in relevant units (default is 1.0).

    Returns:
        float: The total emissions from the process.

    Raises:
        ValueError: If no emission factor is found for the specified category, origin, or source.
    """
    try:
        # Define the label for emission factor search
        label_use = f"{category}_{origen}_{source}"

        # Look up emission factor
        emission_factor = func_lca_get_emission_factor(label_category=category, label_use=label_use)

        # Calculate emissions from process amount and emission factor
        emission_process = amount * emission_factor

        return emission_process
    except ValueError as e:
        # Provide detailed error message if emission factor lookup fails
        raise ValueError(
            f"Failed to calculate emissions for category '{category}', origin '{origen}', "
            f"source '{source}', and amount {amount}. Error: {e}"
        )

# get cumulated emissions for plastic processes (drying, regranulation)
def func_lca_emissions_processes_plastic(process: str, df: pd.DataFrame, source_electricity: str = "mix", source_heat: str = "mix") -> float:

    # Initialize values for process emissions in kg CO2e per kWh
    emission_process_electricity = 0 
    emission_process_heat = 0 
    relevant_category = "plastic"

    # filter dataframe for relevant category
    category_share, df_filtered = func_lca_get_category_share(df=df, relevant_category=relevant_category)
     
    # set labels depending on process
    match process:
        case "drying":
            column_label_electricity = "drying_electricity_use_kWh_per_kg"
            column_label_heat = "drying_heat_use_kWh_per_kg"

        case "regranulation":
            column_label_electricity = "regranulation_electricity_use_kWh_per_kg"
            column_label_heat = "regranulation_heat_use_kWh_per_kg"

    # for every material fraction
    for k in range(df_filtered.shape[0]):
        
        # get material name and percentage from df_result in a list [name, percentage]
        material_type = df_filtered.iloc[k][label_material_type]
        material_share = df_filtered.iloc[k][label_material_share] / 100

        # lookup electricity use for each process from list_material
        electricity_use = float(df_materials.loc[df_materials["abbreviation"].str.fullmatch(material_type, case=False, na=False), column_label_electricity])
        heat_use = float(df_materials.loc[df_materials["abbreviation"].str.fullmatch(material_type, case=False, na=False), column_label_heat])

        # calculate emissions [kg CO2e/kg/share plastic] for the relevant process from energy use [kWh], material share [-] and emission factor [kg CO2/kWh]
        emission_process_electricity += electricity_use * (material_share/category_share) * func_lca_emissions_process(category="electricity", origen=lca_emission_origen, source=source_electricity)
        emission_process_heat += heat_use * (material_share/category_share) * func_lca_emissions_process(category="heat", origen=lca_emission_origen, source=source_heat)

    # add up both emission sources
    emission_process = emission_process_electricity + emission_process_heat
    return emission_process

# get cumulated emissions for melting process
def func_lca_emissions_process_melting(df: pd.DataFrame) -> float:
    
    # initialize variable for process emissions
    emission_process = 0
    relevant_category = "metal"

    # filter dataframe for relevant category
    category_share, df_filtered = func_lca_get_category_share(df=df, relevant_category=relevant_category)
    
    # loop through every type of material
    for k in range(df_filtered.shape[0]):

        # get material type, category and share from df_result
        material_type = df_filtered.iloc[k][label_material_type]
        material_share = df_filtered.iloc[k][label_material_share] / 100

        # calculate and add process emission of material
        emission_process +=  (material_share/category_share) * func_lca_emissions_process(category=f"melting-{relevant_category}", origen=lca_emission_origen, source=material_type)

    return emission_process

# get avoided emissions for secondary materials 1. plastic, 2. metal, depending on material in stream
def func_lca_emissions_avoided_material(df: pd.DataFrame, relevant_category: str) -> float:
    
    # initialize variable for avoided emissions
    emission_avoided_material = 0

    # filter dataframe for relevant category
    category_share, df_filtered = func_lca_get_category_share(df=df, relevant_category=relevant_category)

    # loop through every type of material
    for k in range(df_filtered.shape[0]):

        # get material type, category and share from df_result
        material_type = df_filtered.iloc[k][label_material_type]
        material_share = df_filtered.iloc[k][label_material_share] / 100

        # calculate and add avoided emission of material
        emission_avoided_material +=  (material_share/category_share) * func_lca_emissions_process(category=f"production-{relevant_category}", origen=lca_emission_origen, source=material_type)

    return emission_avoided_material

# calculate material weight per year in ton
def func_lca_get_weight_per_year(weight: float, frequency: str) -> float:
    """
    Calculates the material weight per year in tons based on the given frequency.

    Parameters:
        weight (float): The weight in tons for the specified frequency.
        frequency (str): The frequency of the weight. Expected values:
                         - "Tag" (daily)
                         - "Woche" (weekly)
                         - "Monat" (monthly)
                         - "Quartal" (quarterly)
                         - "Jahr" (annually)

    Returns:
        float: The calculated weight per year in tons.

    Raises:
        ValueError: If the frequency is not recognized.
    """
    # Match the frequency and calculate weight per year
    match frequency:
        case "Tag":
            weight_per_year = weight * 365
        case "Woche":
            weight_per_year = weight * 52
        case "Monat":
            weight_per_year = weight * 12
        case "Quartal":
            weight_per_year = weight * 4
        case "Jahr":
            weight_per_year = weight
        case _:
            bad_input_error("Bitte wähle die Frequenz des Abfalls aus.", page="subpages/input/product_further.py")

    return weight_per_year

# function to calculate emissions for scenario: materials go to waste-to-energy-plant
def func_lca_emissions_scenario_wte(df: pd.DataFrame, df_emission: pd.DataFrame) -> pd.DataFrame: 
    
    # determine the new index to add the scenario as new row to the emission dataframe
    new_index = len(df_emission)

    # transport to waste-to-energy(wte)-plant
    df_emission.at[new_index, "transport_wte"] = func_lca_emissions_transport(lca_vehicle_to_wte, lca_distance_to_wte, lca_declared_unit)

    # incineration of waste in waste-to-energy(wte)-plant
    df_emission.at[new_index, "endoflife_incineration"] = func_lca_emissions_incineration(df=df, weight=lca_declared_unit)
    energy_incineration = func_lca_energy_incineration(df=df, weight=lca_declared_unit)

    # sorting process after incineration between metals (recycling) and dross (landfill)
    share_metal, _ = func_lca_get_category_share(df_result, "metal") # share for remaining metal compared to 1 kg material
    share_plastic, _ = func_lca_get_category_share(df_result, "plastic") # share for remaining plastic compared to 1 kg material
    share_dross = lca_share_dross * share_plastic # share for the remaining dross from burned plastic compared to 1 kg waste

    # transport of dross to landfill + enmissions from landfill
    df_emission.at[new_index, "transport_landfill"] = share_dross * func_lca_emissions_transport(lca_vehicle_to_landfill, lca_distance_to_landfill, lca_declared_unit)
    df_emission.at[new_index, "endoflife_landfill"] = share_dross * func_lca_get_emission_factor("landfill", "dross")

    # recycling of metal materials
    # transport to recycling facility
    df_emission.at[new_index, "transport_recycler_metal"] = share_metal * func_lca_emissions_transport(lca_vehicle_to_recycler_metal, lca_distance_to_recycling_metal, lca_declared_unit)

    # sorting, shredding and cleaning of metals at recycling plant
    df_emission.at[new_index, "processing_metal_sorting"] = share_metal * func_lca_emissions_process(category="electricity", amount=lca_electricity_use_sorting_metal, origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "processing_metal_shredding"] = share_metal * func_lca_emissions_process(category="electricity", amount=lca_electricity_use_shredding_metal, origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "processing_metal_cleaning"] = share_metal * sum([
        func_lca_emissions_process(category="electricity", amount=lca_electricity_use_cleaning_metal, origen=lca_emission_origen, source="mix"), 
        func_lca_emissions_process(category="heat", amount=lca_heat_use_cleaning_metal, origen=lca_emission_origen, source="mix"),
        func_lca_emissions_process(category="water", amount=lca_water_use_cleaning_metal, origen="EU", source="groundwater"),
        func_lca_emissions_process(category="wastewater", amount=lca_wastewater_use_cleaning_metal, origen=lca_emission_origen, source="default"),
        func_lca_emissions_process(category="ressource", amount=lca_solvent_use_cleaning_metal, origen=lca_emission_origen, source="cleaning_agent")
    ])

    # melting down metals (depending on metal)
    df_emission.at[new_index, "processing_metal_melting"] = share_metal * func_lca_emissions_process_melting(df=df_result)

    # advantage due to secondary materials and energy generation
    #  Secondary Material (metal)
    df_emission.at[new_index, "avoided_electricity"] = lca_substitution_factor_electricity * -func_lca_emissions_process(category="electricity", amount=energy_incineration[0], origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "avoided_heat"] = lca_substitution_factor_heat * -func_lca_emissions_process(category="heat", amount=energy_incineration[1], origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "avoided_metal"]  = lca_substitution_factor_metal * -func_lca_emissions_avoided_material(df=df_result, relevant_category="metal")

    # Group emissions by category
    df_emission.at[new_index, "processing_metal_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="processing_metal", exclude_substring="total")
    df_emission.at[new_index, "endoflife_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="endoflife", exclude_substring="total")
    df_emission.at[new_index, "transport_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="transport", exclude_substring="total")
    df_emission.at[new_index, "avoided_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="avoided", exclude_substring="total")
    df_emission.at[new_index, "emissions_overall_per_declared_unit"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="total", exclude_substring="-")

    # Total emission per declared unit, by total waste weight (now and per year)
    material_weight_per_year = func_lca_get_weight_per_year(weight=material_weight_regular, frequency=material_frequency)
    
    df_emission.at[new_index, "emissions_overall_per_weight"] = waste_weigth * df_emission.at[new_index, "emissions_overall_per_declared_unit"]
    df_emission.at[new_index, "emissions_overall_per_year"] = material_weight_per_year * df_emission.at[new_index, "emissions_overall_per_declared_unit"]

    return df_emission

# function to calculate emissions for scenario: materials go to recycling
def func_lca_emissions_scenario_recycling(df: pd.DataFrame, df_emission: pd.DataFrame) -> pd.DataFrame: 
    
    # determine the new index to add the scenario as new row to the emission dataframe
    new_index = len(df_emission)

    # transport to recycling plant
    df_emission.at[new_index, "transport_recycler_plastic"] = func_lca_emissions_transport(lca_vehicle_to_recycler_plastic, lca_distance_to_recycler_plastic, lca_declared_unit)

    # recycling of materials
    # sorting, shredding, cleaning, drying and regranulation
    df_emission.at[new_index, "processing_plastic_sorting"] = func_lca_emissions_process(category="electricity", amount=lca_electricity_use_sorting_mixed, origen=lca_emission_origen, source="mix")

    share_metal, _ = func_lca_get_category_share(df_result, "metal") # share for remaining metal compared to 1 kg material
    share_plastic_init, _ = func_lca_get_category_share(df_result, "plastic") # share for remaining plastic compared to 1 kg material
    share_nonrecycable_init = 0 # TODO Abhängig machen zusammen mit share_plastic von den Kunststofffraktionen im Abfall, die nicht recycelbar sind (mit recycling Advancement aus material list, ggf. als weitere Spalte in df_result etablieren)

    # recycling of plastics (shredding, cleaning, drying, regranulation)
    df_emission.at[new_index, "processing_plastic_shredding"] = share_plastic_init * func_lca_emissions_process(category="electricity", amount=lca_electricity_use_shredding_plastic, origen=lca_emission_origen, source="mix")
    share_plastic = share_plastic_init * lca_share_io_shredding
    
    df_emission.at[new_index, "processing_plastic_cleaning"] = share_plastic * sum([
            func_lca_emissions_process(category="electricity", amount=lca_electricity_use_cleaning_plastic, origen=lca_emission_origen, source="mix"), 
            func_lca_emissions_process(category="heat", amount=lca_heat_use_cleaning_plastic, origen=lca_emission_origen, source="mix"),
            func_lca_emissions_process(category="water", amount=lca_water_use_cleaning_plastic, origen="EU", source="groundwater"),
            func_lca_emissions_process(category="wastewater", amount=lca_wastewater_use_cleaning_plastic, origen=lca_emission_origen, source="default"),
            func_lca_emissions_process(category="ressource", amount=lca_solvent_use_cleaning_plastic, origen=lca_emission_origen, source="cleaning_agent")
        ])    
    share_plastic *= lca_share_io_cleaning

    # depending on plastic type
    df_emission.at[new_index, "processing_plastic_drying"] = share_plastic * func_lca_emissions_processes_plastic(process="drying", df=df)
    df_emission.at[new_index, "processing_plastic_regranulation"] = share_plastic * func_lca_emissions_processes_plastic(process="regranulation", df=df)
    share_plastic *= lca_share_io_regranulation
    share_nonrecycable = share_nonrecycable_init + share_plastic_init * (1 - lca_share_io_overall)

    # recycling of metal materials
    # transport to recycling facility
    df_emission.at[new_index, "transport_recycler_metal"] = share_metal * func_lca_emissions_transport(lca_vehicle_to_recycler_metal, lca_distance_to_recycling_metal, lca_declared_unit)

    # sorting, shredding and cleaning of metals at recycling plant
    df_emission.at[new_index, "processing_metal_sorting"] = share_metal * func_lca_emissions_process(category="electricity", amount=lca_electricity_use_sorting_metal, origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "processing_metal_shredding"] = share_metal * func_lca_emissions_process(category="electricity", amount=lca_electricity_use_shredding_metal, origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "processing_metal_cleaning"] = share_metal * sum([
        func_lca_emissions_process(category="electricity", amount=lca_electricity_use_cleaning_metal, origen=lca_emission_origen, source="mix"), 
        func_lca_emissions_process(category="heat", amount=lca_heat_use_cleaning_metal, origen=lca_emission_origen, source="mix"),
        func_lca_emissions_process(category="water", amount=lca_water_use_cleaning_metal, origen="EU", source="groundwater"),
        func_lca_emissions_process(category="wastewater", amount=lca_wastewater_use_cleaning_metal, origen=lca_emission_origen, source="default"),
        func_lca_emissions_process(category="ressource", amount=lca_solvent_use_cleaning_metal, origen=lca_emission_origen, source="cleaning_agent")
    ])

    # melting down metals (depending on metal)
    df_emission.at[new_index, "processing_metal_melting"] = share_metal * func_lca_emissions_process_melting(df=df_result)

    # incineration of non-recycable materials
    # transport to waste-to-energy(wte)-plant
    df_emission.at[new_index, "transport_wte"] = share_nonrecycable * func_lca_emissions_transport(lca_vehicle_to_wte, lca_distance_to_wte, lca_declared_unit)

    # incineration of waste in waste-to-energy(wte)-plant
    df_emission.at[new_index, "endoflife_incineration"] = share_nonrecycable * func_lca_emissions_incineration(df=df, weight=lca_declared_unit)
    energy_incineration = [x * share_nonrecycable for x in func_lca_energy_incineration(df=df, weight=lca_declared_unit)]
    share_dross = lca_share_dross * share_nonrecycable # share for the remaining dross compared to 1 kg waste

    # transport of dross to landfill + enmissions from landfill
    df_emission.at[new_index, "transport_landfill"] = share_dross * func_lca_emissions_transport(lca_vehicle_to_landfill, lca_distance_to_landfill, lca_declared_unit)
    df_emission.at[new_index, "endoflife_landfill"] = share_dross * func_lca_get_emission_factor("landfill", "dross")

    # advantage due to secondary materials and energy generation
    df_emission.at[new_index, "avoided_electricity"] = lca_substitution_factor_electricity * -func_lca_emissions_process(category="electricity", amount=energy_incineration[0], origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "avoided_heat"] = lca_substitution_factor_heat * -func_lca_emissions_process(category="heat", amount=energy_incineration[1], origen=lca_emission_origen, source="mix")
    df_emission.at[new_index, "avoided_plastic"]  = lca_substitution_factor_plastic * -func_lca_emissions_avoided_material(df=df, relevant_category="plastic")
    df_emission.at[new_index, "avoided_metal"]  = lca_substitution_factor_metal * -func_lca_emissions_avoided_material(df=df, relevant_category="metal")

    # Group emissions by category
    df_emission.at[new_index, "processing_plastic_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="processing_plastic", exclude_substring="total")
    df_emission.at[new_index, "processing_metal_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="processing_metal", exclude_substring="total")
    df_emission.at[new_index, "endoflife_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="endoflife", exclude_substring="total")
    df_emission.at[new_index, "transport_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="transport", exclude_substring="total")
    df_emission.at[new_index, "avoided_total"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="avoided", exclude_substring="total")
    df_emission.at[new_index, "emissions_overall_per_declared_unit"] = func_sum_columns_with_conditions(df=df_emission, row_label=new_index , include_substring="total", exclude_substring="-")

    # Total emission per declared unit, by total waste weight (now and per year)
    material_weight_per_year = func_lca_get_weight_per_year(weight=material_weight_regular, frequency=material_frequency)
    
    df_emission.at[new_index, "emissions_overall_per_weight"] = waste_weigth * df_emission.at[new_index, "emissions_overall_per_declared_unit"]
    df_emission.at[new_index, "emissions_overall_per_year"] = material_weight_per_year * df_emission.at[new_index, "emissions_overall_per_declared_unit"]

    return df_emission

# compare emission values
def func_lca_compare_emissions(val_recycling: float, val_wte: float) -> tuple:
    """
    Compares emissions between recycling and waste incineration (WTE) and returns 
    both the difference and a sentence describing the comparison.

    Parameters:
        val_recycling (float): Emissions for recycling in kg.
        val_wte (float): Emissions for waste incineration in kg.

    Returns:
        tuple: A tuple containing:
            - float: The calculated difference in emissions (recycling perspective).
            - str: A sentence describing the comparison.
    """
    # Calculate the difference
    difference = val_wte - val_recycling
    
    # Create a sentence based on the difference
    if difference > 0:
        sentence = f"Die Emissionen für das Recycling sind {difference:.2f} t niedriger als für die Abfallverbrennung."
    elif difference < 0:
        sentence = f"Die Emissionen für das Recycling sind {-difference:.2f} t höher als für die Abfallverbrennung."
    else:
        sentence = "Die Emissionen für das Recycling und die Abfallverbrennung sind identisch."
    
    return difference, sentence

## Main script
# Step 1: Check hazourdous/non-hazourdous status via REACH-conformity
# 1.1 If waste is not conform to REACH, categorize as "Sondermüll"
if reach_conformity == "Nein":
    material_score = ws_boundary_min
   
# Step 2: Check if fractions are for the most part potentially recycable
# 2.1 Filter materials dataframe for recycling advancement
df_materials_recyAdvance = df_materials[df_materials.recycling_advance > 0] #filter dataframe for all materials with the potential for recycling >0
decision_tree_mats = df_materials_recyAdvance.abbreviation.tolist()

# 2.2 Adding up all
assert len(material_type) == material_fraction_number, "Length of material_type and material_fraction_number must be equal."
for k in range(material_fraction_number):
  # Check if the kth entry of material_type is in a list of materials of dataframe
    if material_type[k] in df_materials_recyAdvance.abbreviation.tolist():
        # Add the kth value of material_share to percent_valuable_substance
        percent_valuable_substance += material_share[k]/100

if percent_valuable_substance < set_threshold_assessibility:
    material_score = ws_boundary_g_f

# Step 3: Check if possible to sort fractions by different methods
# Initialize dataframes for output scores (1. for sorting, 2. for final results)
df_result = func_initializeResultDf(amount=material_fraction_number, label_columns=columns_materials)
value_medium_clean_sort = ws_boundary_d_c + (ws_boundary_c_b - ws_boundary_d_c)/2 

# if waste consists only of one fraction, no sorting is necessary
if material_fraction_number == 1: 
    material_score = value_medium_clean_sort #Einteilung als Recyling, werkstofflich, sortenrein

    # result_sorting in Dataframe = 1
    df_result.at[0, label_material_result_sorting] = 1

# if waste consists of more than 1 fraction, sorting is necessary
elif material_fraction_number > 1 and material_fraction_number <= 5:
  
    # Store dataframes in list to use in loop
    list_df_sort = [df_sort_ferromagnetic, df_sort_eddycurrent, df_sort_density, df_sort_electrostatic, df_sort_sensorbased]
    list_sort_weight = [wt_ferromagnetic, wt_eddycurrent, wt_density, wt_electrostatic, wt_sensorbased]

    # Initialize dataframes for output scores (1. for sorting)
    df_result_sorting = func_initializeOutputDf(amount = material_fraction_number, label_columns=columns_sorting, material_type=material_type)

    # loop through each sorting method and obtain values for all material pairing
    for k in range(len(list_df_sort)):

        # Obtain values from function and add to dataframe
        df_result_sorting[columns_sorting[k+1]] = func_checkSorting(amount=material_fraction_number, df=list_df_sort[k])

    # Check if one material can be sorted completly from any other (= 1, sortenrein)
    for k in range(material_fraction_number):

        # Get material string 
        material_name = material_type[k]

        # Check which row contains material name
        matching_row_indices = func_get_indices_for_material(df=df_result_sorting, column=label_material_pairing, material=material_name)
       
        # Check if all relevant rows for a material contain at least a 1
        value_to_find = 1
        list_check_clear_sort = []

        for l in range(len(matching_row_indices)):
            contains_value = value_to_find in df_result_sorting.iloc[matching_row_indices[l], :].values
            list_check_clear_sort.append(contains_value)

        # if the material can be sorted from other materials: 
        if all(list_check_clear_sort) == True:
            # res_sort = 1
            df_result.at[k, label_material_result_sorting] = 1

        # if the material can NOT be sorted completly from other materials:  
        elif all(list_check_clear_sort) == False:
            # res_sort = weigthed average of all results by sorting method for specific material
            for l in range(len(list_check_clear_sort)):
                
                # get material pairing which cannot be sorted completly (!=1)
                if list_check_clear_sort[l] == False:
                    # extract values from dataframe column

                    values_to_average = df_result_sorting.iloc[matching_row_indices[l],1:].tolist()
                    # multiply each value with the corresponding weight
                    weighted_values_to_average = [a * b for a, b in zip(values_to_average, list_sort_weight)]
                    # calculate mean and store in result dataframe
                    df_result.at[k, label_material_result_sorting] = sum(weighted_values_to_average) / len(weighted_values_to_average)

    # get the weighted mean of the material specific results of the sorting options
    weighted_mean = func_calculate_weighted_mean(df=df_result, value_col=label_material_result_sorting, weight_col=label_material_share)

    # get the preliminary material score
    value_medium_clean_sort = ws_boundary_d_c + (ws_boundary_c_b - ws_boundary_d_c)/2 
    material_score = func_map_to_scale(value=weighted_mean, return_if_1=value_medium_clean_sort, start_scale=ws_boundary_e_d, end_scale=ws_boundary_d_c)
    
    # consider compatibility and give mali, if not compatible
    # Obtain values from function and add to result dataframe for sorting
    df_result_sorting[label_sort_compatibility] = func_checkSorting(amount=material_fraction_number, df=df_sort_compatibility)

    # for each material: calculate the mean value for comparability from pairings containing the material
    df_result[label_material_result_compatibility] = func_calculate_mean_for_materials(df=df_result_sorting, materials=material_type, material_col=label_material_pairing, value_col=label_sort_compatibility)

    # calculate the total mean value for comparability from the values of each material weighted with the material share
    comparability_total = func_calculate_weighted_mean(df=df_result, value_col=label_material_result_compatibility, weight_col=label_material_share) 

    # map total comparability to scale to determine the malus given
    comparability_malus = func_map_to_scale(value=comparability_total, return_if_1=comparability_malus_min, start_scale=comparability_malus_max, end_scale=comparability_malus_min)

def parse_company_materials(materials: str) -> List[str]:
    return list(map(lambda x: x.strip(), materials.split(',')))

def company_recycles(comp_mats: List[str], target_mat: str) -> bool:
    comp_mat = map_material(target_mat, 'decision_tree', 'companies.csv')
    if comp_mat is None:
        return False
    return any(map(lambda x: x in comp_mat, comp_mats))

def score_material(materials: str) -> int:
    mats = list(map(lambda x: x.strip(), materials.split(',')))
    score = 0
    for user_mat in material_type:
        if company_recycles(mats, user_mat):
            score += 1
    return score

if 'coordinates_data' not in st.session_state:
    bad_input_error("Der Standort des Unternehmens ist uns unbekannt, bitte überprüfe deine Eingabe.", 'subpages/input/company.py')
coords: Tuple[None | float, None | float] = st.session_state.coordinates_data["latitude"], st.session_state.coordinates_data["longitude"]
user_lat, user_lon = coords

if user_lat is None or user_lon is None:
    bad_input_error("Der Standort des Unternehmens ist uns unbekannt, bitte überprüfe deine Eingabe.", 'subpages/input/company.py')

def score_distance(row):
    from math import radians, sin, cos, acos
    assert user_lat is not None and user_lon is not None

    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        return 6371.01 * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    return calculate_distance(user_lat, user_lon, row['lat'], row['lon'])

def recommend() -> pd.DataFrame:
    scores = companies['materials'].apply(score_material)
    scored_companies = companies.copy()
    scored_companies['distance'] = scored_companies.apply(score_distance, axis=1)
    scored_companies['match_score'] = scores + 10 / (scored_companies['distance'] + 100)
    res = scored_companies.sort_values(by='match_score', ascending=False)
    return res[:3] # type: ignore


recommendations = recommend()

# consider contamination and give malus, if materials are contaminated
contamination_malus = func_calculate_contamination_malus(material_contamination=material_contamination, scale_contamination_malus_min=contamination_malus_min, scale_contamination_malus_max=contamination_malus_max)

# update material score
material_score += (comparability_malus + contamination_malus)

# get the material score category and output sentence
ws_category, ws_sentence = func_evaluateWS(material_score)

## get location data from Recycler #TODO WeSort: Hier den Algorithmus zur Verknüpfung mit dem Recycler einfügen. 
# Habe bereits die ermittelten Daten zu Längen- und Breitengrad des Abfallursprungs als list bereitgestellt. 
# Als Output sollte u.a. die Transportdistanz vom Unternehmen zum Recycler (lca_distance_to_recycler) angegeben werden. Diese wird unten für die life cycle analysis benötigt.
#company_coordinates = st.session_state.coordinates_data

lca_distance_to_recycler_plastic = list(recommendations['distance'])[0]

## life cycle analysis (lca) and comparison of current and proposed waste treatment
# Initialize dataframe for calculated emissions
df_emission = pd.DataFrame(columns = columns_emissions)

# Scenario 1: Calulation of lca for current waste treatment (assumption waste-to-energy)
# Use function to calculate emissions in this scenario
df_emission = func_lca_emissions_scenario_wte(df=df_result, df_emission=df_emission)

# Scenario 2. Calculation of lca of future waste treatment (recycling)
df_emission = func_lca_emissions_scenario_recycling(df=df_result, df_emission=df_emission)

# Compare emissions from both scenario
emission_recyling_per_weight = df_emission.at[1, "emissions_overall_per_weight"] # emission in t CO2 / t material
emission_wte_per_weight = df_emission.at[0, "emissions_overall_per_weight"] # emission in t CO2 / t material
emission_comparison_per_weight, emission_sentence_per_weight = func_lca_compare_emissions(df_emission.at[1, "emissions_overall_per_weight"], df_emission.at[0, "emissions_overall_per_weight"])
emission_comparison_per_year, emission_sentence_per_year = func_lca_compare_emissions(df_emission.at[1, "emissions_overall_per_year"], df_emission.at[0, "emissions_overall_per_year"])

# Title Section
st.title("PlastIQ")
st.subheader("Dein Wertstoff-Score (WS)")

# Main Content
col1, col2 = st.columns([1, 2])  # Adjust proportions to balance text and map

# Left column: Score and recommendation
with col1:
    st.metric(label="WS", value=f"{round(material_score, 1)} %", help="Der Wertstoff-Score wurde auf Basis deiner Eingaben ermittelt. Er gibt an, welche Verwertungsmethode dafür geeignet erscheint.")
    st.write(f"Dein Wertstoff erreicht den Score {ws_category}. {ws_sentence}")

    if ws_category in ['E', 'F', 'G', 'H']:
        st.warning("Wahrscheinlich ist es nicht recycelbar, aber du kannst trotzdem anfragen")

    st.write("#### Wir empfehlen dir folgende Recyclers:")


import pydeck as pdk
import numpy as np


for i in range(len(recommendations)):
    rec = recommendations.iloc[i]
    company_materials = parse_company_materials(rec['materials'])
    can_recycle_for_user = list(filter(lambda m: company_recycles(company_materials, m), material_type))
    col1, col2 = st.columns([1, 1])  # Adjust proportions to balance text and map
    with col1:
        # Recycler Recommendation
        st.write(f"### #{i+1}: {rec['company']}")
        st.write(f"Dies Unternehmen recycelt für dich: **{', '.join(can_recycle_for_user)}**")
        st.write(f"Dies Unternehmen recycelt auch: {', '.join(company_materials)}")
        if not np.isnan(rec['volume']):
            st.write(f"Kapazität: {str(rec['volume'])} Tonnen/Jahr")
        else:
            st.write("Kapazität ist uns unbekannt")

        st.write(f"Die Anlage ist **{int(rec['distance'])} km** von dir")
        st.write("Adresse: " + str(rec['address']))
        if rec['phone'] and not pd.isna(rec['phone']):
            st.write("Telefonnummer: " + str(rec['phone']))
        if rec['email'] and not pd.isna(rec['email']):
            st.write("E-Mail: " + str(rec['email']))
        st.write("Website: " + str(rec['website']))
        st.write()

    if rec['lat'] is not None:
        with col2:
            data = pd.DataFrame({
                'lat': [rec['lat'], user_lat],  # Latitude of the points
                'lon': [rec['lon'], user_lon],  # Longitude of the points
                'label': [rec['company'], 'Deine Anlage'],  # Labels
                'color': [[0, 255, 128], [255, 0, 255]]  # Green and Blue in RGB
            })

            # Scatterplot layer for points
            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=10000,  # Radius of the points
                radius_min_pixels=10,
                radius_max_pixels=10,
                pickable=True
            )

            # Text layer for labeling points directly on the map
            text_layer = pdk.Layer(
                "TextLayer",
                data,
                get_position=["lon", "lat"],
                get_text="label",
                get_color=[0, 0, 0],  # White text for better visibility
                get_size=24,  # Font size
                get_alignment_baseline="'bottom'",  # Align text to bottom of the point
            )

            # Define the map view centered at Germany
            view_state = pdk.ViewState(latitude=51.1657, longitude=10.4515, zoom=5, pitch=0)

            # Render the map with both layers (points + text)
            st.pydeck_chart(pdk.Deck(
                layers=[point_layer, text_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9"  # Light theme
            ))
    st.divider()  # Horizontal divider

col3, col4 = st.columns([1,1])

# Ecological Section
with col3:
    st.header("Ökologisch", help="""
Die Ökobilanz vergleicht die Treibhausgasemissionen (THG) zweier Szenarien in der Einheit Tonne CO2-Äquivalent pro Tonne Wertstoff oder kurz t CO2/t. Als erstes Szenario wird die Verbrennung des Abfalls in einer Müllverbrennungsanlage mit Rückgewinnung und anschließender Nutzung von Wärme und elektrischer Energie verglichen. Als zweites Szenario wird das Recycling der Wertstoffe betrachtet, d.h. die Aufbereitung zu Rezyklaten, die in späteren Produkten eingesetzt werden können.

Für die Berechnung werden die Emissionen aus den Prozessen, dem Transport und der Verbrennung der Stoffe, die am Ende des Lebenszyklus der Wertstoffe anfallen, berücksichtigt. Mögliche Vorteile außerhalb der Systemgrenzen durch erzeugte Energie oder aufbereitete Rezyklate, die Neupolymere in Produkten ersetzen können, werden nicht berücksichtigt.
    """)
    st.write(f"Deine Wertstoffmenge: **{waste_weigth} t**")
    st.write(f"THG-Emission Entsorgung bisher (geschätzt): **{round(emission_wte_per_weight,1)} t CO₂e/t**")
    st.write(f"THG-Emission Recycling (erwartet): **{emission_recyling_per_weight.round(1)} t CO₂e/t**")
    st.subheader(f"Einsparung: **{emission_comparison_per_weight.round(1)} t CO₂e**")
    st.write(emission_sentence_per_weight)

    df_thg_emissions = df_emission[['endoflife_total', 'transport_total']]
    df_thg_emissions['processing_total'] = df_emission['processing_plastic_total'] + df_emission['processing_metal_total']

    # import plotly.graph_objects as go
    # categories = ['End of Life', 'Transport', 'Processing']
    # fig = go.Figure()
    # fig.add_trace(go.Bar(name='Verbrennung', x=categories, y=df_thg_emissions.loc[0].values))
    # fig.add_trace(go.Bar(name='Recycling', x=categories, y=df_thg_emissions.loc[1].values))
    # fig.update_layout(barmode='stack', title='Emissionen nach Schritt', xaxis_title='', yaxis_title='t CO2e')
    # st.plotly_chart(fig)

    df_thg_emissions = df_thg_emissions * waste_weigth

    import plotly.graph_objects as go
    categories = ['Verbrennung', 'Recycling']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Aufbereitung', x=categories, y=df_thg_emissions['endoflife_total']))
    fig.add_trace(go.Bar(name='Transport', x=categories, y=df_thg_emissions['transport_total']))
    fig.add_trace(go.Bar(name='Verwertung', x=categories, y=df_thg_emissions['processing_total']))
    fig.update_layout(barmode='stack', title='Emissionen: Verbrennung vs Recycling', xaxis_title='', yaxis_title='t CO2e')
    st.plotly_chart(fig)

# Economical Section
with col4:
    st.header("Ökonomisch")
    def display_bars(title: str, weight: float, warn_msg: str) -> None:
        res = {}
        sum = 0.0
        for (mat, share) in zip(material_type, material_share):
            w = weight * share
            res[mat] = w
            sum += w
        st.write(f'### {title}')
        if math.isclose(weight, 0.0) or len(res) == 0:
            bad_input_warning(warn_msg, 'subpages/input/product_further.py')
        else:
            st.write(f"Gesamt: {round(sum, 1)} €")
            st.bar_chart(res)

    display_bars("Wert der verfügbaren Wertstoffe", waste_weigth, "Bitte gebe die verfügbare Wertstoffmenge an")
    display_bars("Wert der Wertstoffe pro " + material_frequency, material_weight_regular, "Bitte gebe die regelmäßige Wertstoffmenge an")
