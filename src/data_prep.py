import pandas as pd

def load_and_prep_data(dframes):
    ## Load data
    print("Loading Dataframes...")
    for key, value in dframes.items():
        globals()[key] = pd.read_csv(value)
        print(f"{key} loaded successfully...")
    
    ## Deal with multiple types
    print("Dealing with mixed types in person_df, accident_df & vehicle_df...")
    person_df[['SEX', 'AGE', 'SEATING_POSITION', 'Inj Level Desc']] = person_df[['SEX', 'AGE', 'SEATING_POSITION', 'Inj Level Desc']].astype(str)
    accident_df[['POLICE_ATTEND', 'SEVERITY']] = accident_df[['POLICE_ATTEND', 'SEVERITY']].astype(str)
    vehicle_df[['VEHICLE_YEAR_MANUF', 'INITIAL_DIRECTION', 'FINAL_DIRECTION', 'CAUGHT_FIRE', 'LAMPS']] = vehicle_df[['VEHICLE_YEAR_MANUF', 'INITIAL_DIRECTION', 'FINAL_DIRECTION', 'CAUGHT_FIRE', 'LAMPS']].astype(str)

    ## convert datetime col to datetime format
    print("Converting accidentdate and accidenttime to relevant datetimeformat...")
    accident_df['ACCIDENTDATE'] = pd.to_datetime(accident_df['ACCIDENTDATE'], dayfirst=True)
    accident_df['ACCIDENTTIME'] = pd.to_datetime(accident_df['ACCIDENTTIME'], errors='coerce')

    ## the columns we are choosing to include in our modeling
    print("Choosing columns...")
    accident_columns = ['ACCIDENT_NO', 'ACCIDENTDATE', 'ACCIDENTTIME', 'Accident Type Desc', 'Day Week Description', 
                        'DCA Description', 'Light Condition Desc', 'NODE_ID', 'NO_OF_VEHICLES', 'NO_PERSONS', 
                        'POLICE_ATTEND', 'Road Geometry Desc', 'SEVERITY', 'SPEED_ZONE']
    event_columns = ['ACCIDENT_NO', 'Event Type Desc', 'VEHICLE_1_ID', 'Vehicle 1 Coll Pt Desc', 
                     'VEHICLE_2_ID', 'Vehicle 2 Coll Pt Desc', 'PERSON_ID', 'OBJECT_TYPE', 'Object Type Desc']
    location_columns = ['ACCIDENT_NO', 'NODE_ID', 'ROAD_TYPE', 'ROAD_TYPE_INT', 'DISTANCE_LOCATION', 'DIRECTION_LOCATION']
    atmospheric_columns = ['ACCIDENT_NO', 'Atmosph Cond Desc']
    node_columns = ['ACCIDENT_NO', 'NODE_ID', 'NODE_TYPE', 'LGA_NAME', 'REGION_NAME', 'DEG_URBAN_NAME']
    node_complex_columns = ['ACCIDENT_NO', 'NODE_ID', 'COMPLEX_INT_NO']
    person_columns = ['ACCIDENT_NO', 'PERSON_ID', 'VEHICLE_ID', 'SEX', 'AGE', 'SEATING_POSITION', 'HELMET_BELT_WORN', 'Road User Type Desc', 'LICENCE_STATE', 
                      'PEDEST_MOVEMENT', 'EJECTED_CODE', 'Inj Level Desc']
    road_columns = ['ACCIDENT_NO', 'Surface Cond Desc']
    vehicle_columns = ['ACCIDENT_NO', 'VEHICLE_ID', 'VEHICLE_YEAR_MANUF', 'VEHICLE_DCA_CODE', 'INITIAL_DIRECTION', 
                       'Road Surface Type Desc', 'REG_STATE', 'VEHICLE_BODY_STYLE', 'VEHICLE_MAKE', 'VEHICLE_MODEL', 
                       'Vehicle Type Desc', 'CONSTRUCTION_TYPE', 'FUEL_TYPE', 'NO_OF_WHEELS', 'NO_OF_CYLINDERS', 
                       'SEATING_CAPACITY', 'TARE_WEIGHT', 'TOTAL_NO_OCCUPANTS', 'CARRY_CAPACITY', 'CUBIC_CAPACITY', 
                       'FINAL_DIRECTION', 'DRIVER_INTENT', 'VEHICLE_MOVEMENT', 'TRAILER_TYPE', 'VEHICLE_COLOUR_1', 
                       'VEHICLE_COLOUR_2', 'CAUGHT_FIRE', 'INITIAL_IMPACT', 'LAMPS', 'Traffic Control Desc']
    print("Chose columns successfully...")
    accident_data = accident_df[accident_columns]
    event_data = accident_event_df[event_columns]
    location_data = accident_location_df[location_columns]
    atmospheric_data = atmospheric_cond_df[atmospheric_columns]
    node_data = node_df[node_columns]
    node_complex_data = node_id_complex_int_id_df[node_complex_columns]
    person_data = person_df[person_columns]
    road_data = road_surface_cond_df[road_columns]
    vehicle_data = vehicle_df[vehicle_columns]

    ## Get rid of blank strings and leading/trailing spaces
    print("Snail trail be gone!!")
    for df in [accident_data, event_data, location_data, atmospheric_data, node_data, node_complex_data, person_data, road_data, vehicle_data]:
        df = df.replace('^\s$', None, regex=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    print("Successfully Loaded and Prepped Data...")

    return {
        'accident_data': accident_data,
        'event_data': event_data,
        'location_data': location_data,
        'atmospheric_data': atmospheric_data,
        'node_data': node_data,
        'node_complex_data': node_complex_data,
        'person_data': person_data,
        'road_data': road_data,
        'vehicle_data': vehicle_data
    }



