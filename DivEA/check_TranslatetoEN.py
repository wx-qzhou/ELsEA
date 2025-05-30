# -*- coding: utf-8 -*-
import re
import os
import argparse
from os.path import join
from divea.dataload import read_tab_lines
from Unsuper.TranslatetoEN.translate_data import load_json, dump_json, translate_to_english

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="1m/", required=False, help="input dataset name")
    parser.add_argument('--data_dir', type=str, default="../data/", required=False, help="input dataset file directory")
    parser.add_argument('--kgids', type=str, default="en,de", help="separate two ids with comma. e.g. `fr,en`")
    args = parser.parse_args()

    return args

error_list = ['http://ja.dbpedia.org/resource/Kass_prefecture.',
    'http://ja.dbpedia.org/resource/Cornwall',
    'http://ja.dbpedia.org/resource/Krasnodar',
    "http://ja.dbpedia.org/resource/I'm_going_to_sing_it_for_you.",
    'http://ja.dbpedia.org/resource/(Music)',
    'http://ja.dbpedia.org/resource/UFC_Fight_Night:_Mahida_vs.',
    'http://ja.dbpedia.org/resource/Osama.',
    'http://ja.dbpedia.org/resource/(Applause)',
    'http://ja.dbpedia.org/resource/Haranovsk',
    'http://ja.dbpedia.org/resource/The_Olympic_Stadium.',
    'http://ja.dbpedia.org/resource/Greek',
    'http://ja.dbpedia.org/resource/Caine_of_Magic.',
    'http://ja.dbpedia.org/resource/Gorge.',
    'http://ja.dbpedia.org/resource/Cygnus',
    'http://ja.dbpedia.org/resource/Adams_County.',
    'http://ja.dbpedia.org/resource/UFC_Fight_Night:_Biggramt_vs.',
    'http://ja.dbpedia.org/resource/American_Idiot.',
    'http://ja.dbpedia.org/resource/Carol_County.',
    'http://ja.dbpedia.org/resource/Aquarius',
    'http://ja.dbpedia.org/resource/ETHIOPIA',
    'http://ja.dbpedia.org/resource/Oaxaca',
    'http://ja.dbpedia.org/resource/Cyclopædia',
    "http://ja.dbpedia.org/resource/I'm_going_to_show_you_how_to_do_this.",
    'http://ja.dbpedia.org/resource/Gilbert_Elliott_Marley_Kinin_Maundo.',
    'http://ja.dbpedia.org/resource/Sakura',
    'http://ja.dbpedia.org/resource/Green_County,_U.S.A.',
    'http://ja.dbpedia.org/resource/Armenian',
    'http://ja.dbpedia.org/resource/Arabic',
    'http://ja.dbpedia.org/resource/Republic_of_Crimea',
    'http://ja.dbpedia.org/resource/Colombo',
    'http://ja.dbpedia.org/resource/UFC_101',
    'http://ja.dbpedia.org/resource/Wigur',
    'http://ja.dbpedia.org/resource/PHILIPPINES',
    'http://ja.dbpedia.org/resource/President_of_the_United_States_of_America.',
    'http://ja.dbpedia.org/resource/(Laughter)',
    'http://ja.dbpedia.org/resource/INDONESIA',
    "http://ja.dbpedia.org/resource/Let's_do_another_one.",
    'http://ja.dbpedia.org/resource/UFC_15',
    'http://ja.dbpedia.org/resource/Uppsala',
    'http://ja.dbpedia.org/resource/Its_onley_Rockon_Roll',
    'http://ja.dbpedia.org/resource/&gt;&gt;Hugo_Barra:',
    'http://ja.dbpedia.org/resource/UFC_Fight_Night:_Edgar_vs.',
    'http://ja.dbpedia.org/resource/AUSTRIA',
    'http://ja.dbpedia.org/resource/Azerbaijan',
    'http://ja.dbpedia.org/resource/Mm-hmm.',
    'http://ja.dbpedia.org/resource/Osage_County.',
    'http://ja.dbpedia.org/resource/Subtitles:_@marlonrock1986_(^^V^^)',
    'http://ja.dbpedia.org/resource/Kannada',
    'http://ja.dbpedia.org/resource/Gujarati',
    "http://ja.dbpedia.org/resource/I'm_a_soccer_champion.",
    'http://ja.dbpedia.org/resource/(For_fully_formatted_text,_see_publication)',
    "http://ja.dbpedia.org/resource/So_let's_see_if_we_can_do_that.",
    'http://ja.dbpedia.org/resource/Anaber.',
    'http://ja.dbpedia.org/resource/Ache',
    "http://ja.dbpedia.org/resource/I_don't_know_what_you're_talking_about.",
    'http://ja.dbpedia.org/resource/Avery.',
    'http://ja.dbpedia.org/resource/Aramaic',
    'http://ja.dbpedia.org/resource/Oriya',
    'http://ja.dbpedia.org/resource/Gorillas.',
    'http://ja.dbpedia.org/resource/Urdu',
    'http://ja.dbpedia.org/resource/Green_County.',
    "http://ja.dbpedia.org/resource/I'm_going_to_sing_a_song_for_you.",
    'http://ja.dbpedia.org/resource/Cycling',
    'http://ja.dbpedia.org/resource/Aragon',
    'http://ja.dbpedia.org/resource/(Music)_(Music)',
    "http://zh.dbpedia.org/resource/I_don't_know_what_you're_talking_about.",
    "http://zh.dbpedia.org/resource/I'm_sorry._I'm_sorry.",
    "http://zh.dbpedia.org/resource/Three_gorges",
    "http://zh.dbpedia.org/resource/I'll_be_back_in_a_minute",
    "http://zh.dbpedia.org/resource/Chinese_Japanese",
    "http://zh.dbpedia.org/resource/China_National_Youth_Football_Team",
    "http://zh.dbpedia.org/resource/Goddammit!",
    "http://zh.dbpedia.org/resource/Armenian",
    "http://zh.dbpedia.org/resource/Kyouta_Railway",
    "http://zh.dbpedia.org/resource/I_don't_know_what_you're_talking_about.",
    "http://zh.dbpedia.org/resource/Birmingham.",
    "http://zh.dbpedia.org/resource/Berkeley.",
    "http://zh.dbpedia.org/resource/Sato-san.",
    "http://zh.dbpedia.org/resource/Ohio.",
    "http://zh.dbpedia.org/resource/Pseudo-linguistic",
    "http://zh.dbpedia.org/resource/Sinhala",
    "http://zh.dbpedia.org/resource/Xiangzhou_City",
    "http://zh.dbpedia.org/resource/Yunjian.",
    "http://zh.dbpedia.org/resource/The_All-American_Supermodels_New_Show.",
    "http://zh.dbpedia.org/resource/Beijing",
    "http://zh.dbpedia.org/resource/North_Vancouver.",
    "http://zh.dbpedia.org/resource/Gujarati",
    'de_http://de.dbpedia.org/resource/California',
    'de_http://de.dbpedia.org/resource/Acoustic_guitar',
    'de_http://de.dbpedia.org/resource/Hawaiian_guitar',
    'de_http://de.dbpedia.org/resource/Death',
    'de_http://de.dbpedia.org/resource/Ears',
    'de_http://de.dbpedia.org/resource/Belgium',
    'de_http://de.dbpedia.org/resource/Down_(Band)',
    'de_http://de.dbpedia.org/resource/Cold',
    'de_http://de.dbpedia.org/resource/Olomouc',
    'de_http://de.dbpedia.org/resource/Slovakia',
    'de_http://de.dbpedia.org/resource/Corvette',
    'de_http://de.dbpedia.org/resource/Car_manufacturers',
    'de_http://de.dbpedia.org/resource/Dates',
    'de_http://de.dbpedia.org/resource/Save',
    'de_http://de.dbpedia.org/resource/Zambia',
    'de_http://de.dbpedia.org/resource/Cross-platform',
    'de_http://de.dbpedia.org/resource/Malawi',
    'de_http://de.dbpedia.org/resource/Russia',
    'de_http://de.dbpedia.org/resource/Apache_License',
    'de_http://de.dbpedia.org/resource/Angels',
    'de_http://de.dbpedia.org/resource/Monty_Pythons_Flying_Circus',
    'de_http://de.dbpedia.org/resource/Lot_(river)',
    'de_http://de.dbpedia.org/resource/Reality_TV',
    'de_http://de.dbpedia.org/resource/Mel_Brooks',
    'de_http://de.dbpedia.org/resource/Adventure',
    'de_http://de.dbpedia.org/resource/Wreaths',
    'de_http://de.dbpedia.org/resource/Lviv',
    'de_http://de.dbpedia.org/resource/Latin',
    'de_http://de.dbpedia.org/resource/Courageous',
    'de_http://de.dbpedia.org/resource/Piano',
    'de_http://de.dbpedia.org/resource/The_Circus',
    'de_http://de.dbpedia.org/resource/Czech_Republic',
    'de_http://de.dbpedia.org/resource/Prefecture',
    'de_http://de.dbpedia.org/resource/Flute',
    'de_http://de.dbpedia.org/resource/Poland',
    'de_http://de.dbpedia.org/resource/The_Office',
    'de_http://de.dbpedia.org/resource/Kenya',
    'de_http://de.dbpedia.org/resource/Goodies',
    'de_http://de.dbpedia.org/resource/Second_World_War',
    'de_http://de.dbpedia.org/resource/Cyprus',
    'de_http://de.dbpedia.org/resource/Ethiopia',
    'de_http://de.dbpedia.org/resource/Vistula',
    'de_http://de.dbpedia.org/resource/City',
    'de_http://de.dbpedia.org/resource/Same',
    'de_http://de.dbpedia.org/resource/Black',
    'de_http://de.dbpedia.org/resource/Silence_(film)',
    'de_http://de.dbpedia.org/resource/Main',
    'de_http://de.dbpedia.org/resource/Comedy',
    'de_http://de.dbpedia.org/resource/Second_Chance',
    'de_http://de.dbpedia.org/resource/Chairman',
    'de_http://de.dbpedia.org/resource/Gospel',
    'de_http://de.dbpedia.org/resource/Electropop',
    'de_http://de.dbpedia.org/resource/Egypt',
    'de_http://de.dbpedia.org/resource/World',
    'de_http://de.dbpedia.org/resource/Cartoon',
    'de_http://de.dbpedia.org/resource/Four_devils',
    'de_http://de.dbpedia.org/resource/Spain',
    'de_http://de.dbpedia.org/resource/Source',
    'de_http://de.dbpedia.org/resource/The_Adventure_of_Huck_Finn',
    'de_http://de.dbpedia.org/resource/Fowls_of_the_species_Gallus_domesticus',
    'de_http://de.dbpedia.org/resource/Sleepless_in_New_York',
    'de_http://de.dbpedia.org/resource/Walls',
    'de_http://de.dbpedia.org/resource/Jordan',
    'de_http://de.dbpedia.org/resource/Violin',
    'de_http://de.dbpedia.org/resource/Shots',
    'de_http://de.dbpedia.org/resource/Pigs',
    'de_http://de.dbpedia.org/resource/Zimbabwe',
    'de_http://de.dbpedia.org/resource/Federal_Republic_of_Germany',
    'de_http://de.dbpedia.org/resource/Beavers',
    'de_http://de.dbpedia.org/resource/Seaweed',
    'de_http://de.dbpedia.org/resource/Sports',
    'de_http://de.dbpedia.org/resource/Twins',
    'de_http://de.dbpedia.org/resource/Peaches',
    'de_http://de.dbpedia.org/resource/Hammond_organ',
    'de_http://de.dbpedia.org/resource/Mexico',
    'de_http://de.dbpedia.org/resource/Goal!',
    'de_http://de.dbpedia.org/resource/Lithuania',
    'de_http://de.dbpedia.org/resource/==References==',
    'de_http://de.dbpedia.org/resource/Infidels',
    'de_http://de.dbpedia.org/resource/Scratches',
    'de_http://de.dbpedia.org/resource/Five_Nights_at_Freddy',
    'de_http://de.dbpedia.org/resource/Lead_guitar',
    'de_http://de.dbpedia.org/resource/Small_cars',
    'de_http://de.dbpedia.org/resource/Romance',
    'de_http://de.dbpedia.org/resource/Enslaved',
    'de_http://de.dbpedia.org/resource/Yeah!',
    'de_http://de.dbpedia.org/resource/Laurel_and_Hardy',
    'de_http://de.dbpedia.org/resource/Hercules_(TV_series)',
    'de_http://de.dbpedia.org/resource/Species',
    'de_http://de.dbpedia.org/resource/Customs',
    'de_http://de.dbpedia.org/resource/Madness',
    'de_http://de.dbpedia.org/resource/My_wife,_the_actress',
    'de_http://de.dbpedia.org/resource/Behind_the_Sun',
    'de_http://de.dbpedia.org/resource/The_Commander',
    'de_http://de.dbpedia.org/resource/Saxophone',
    "de_http://de.dbpedia.org/resource/Today_we're_going_to_stroll",
    'de_http://de.dbpedia.org/resource/AI',
    'de_http://de.dbpedia.org/resource/The_millionaire',
    'de_http://de.dbpedia.org/resource/Taipei',
    'de_http://de.dbpedia.org/resource/Tombs',
    'de_http://de.dbpedia.org/resource/Atari_home_computer',
    'de_http://de.dbpedia.org/resource/The_Lady',
    'de_http://de.dbpedia.org/resource/Brno',
    'de_http://de.dbpedia.org/resource/Hungary',
    'de_http://de.dbpedia.org/resource/Art_Rock',
    'de_http://de.dbpedia.org/resource/Other',
    'de_http://de.dbpedia.org/resource/Folk_Rock',
    'de_http://de.dbpedia.org/resource/Turkey',
    'de_http://de.dbpedia.org/resource/German_television',
    'de_http://de.dbpedia.org/resource/Serbia',
    'de_http://de.dbpedia.org/resource/Country',
    'de_http://de.dbpedia.org/resource/Uzbekistan',
    'de_http://de.dbpedia.org/resource/Prague',
    'de_http://de.dbpedia.org/resource/EC',
    'de_http://de.dbpedia.org/resource/Rhine',
    'de_http://de.dbpedia.org/resource/Grassland_languages',
    'de_http://de.dbpedia.org/resource/Tokyo',
    'de_http://de.dbpedia.org/resource/The_Incorrigible',
    'de_http://de.dbpedia.org/resource/Oh,_these_women',
    'de_http://de.dbpedia.org/resource/Norway',
    'de_http://de.dbpedia.org/resource/Rhythm_guitar',
    'de_http://de.dbpedia.org/resource/Tanzania',
    'de_http://de.dbpedia.org/resource/Crime_series',
    'de_http://de.dbpedia.org/resource/Estonia',
    'de_http://de.dbpedia.org/resource/Bulgaria',
    'de_http://de.dbpedia.org/resource/The_Crusaders',
    'de_http://de.dbpedia.org/resource/Owls',
    'de_http://de.dbpedia.org/resource/Mushrooms',
    'de_http://de.dbpedia.org/resource/Katowice',
    'de_http://de.dbpedia.org/resource/Animal',
    'de_http://de.dbpedia.org/resource/Bolivia',
    'de_http://de.dbpedia.org/resource/Trieste',
    'de_http://de.dbpedia.org/resource/The_Border',
    'de_http://de.dbpedia.org/resource/New_Music',
    'de_http://de.dbpedia.org/resource/Giant',
    'de_http://de.dbpedia.org/resource/Sweden',
    'de_http://de.dbpedia.org/resource/Tibet',
    'de_http://de.dbpedia.org/resource/Plovdiv',
    'de_http://de.dbpedia.org/resource/Dominican_Republic',
    'de_http://de.dbpedia.org/resource/List_of_Grafen_and_Markgrafen_of_Provence',
    'de_http://de.dbpedia.org/resource/Total',
    'de_http://de.dbpedia.org/resource/___________________________________________________',
    'de_http://de.dbpedia.org/resource/Christian_Democratic_Union',
    'de_http://de.dbpedia.org/resource/Lore',
    'de_http://de.dbpedia.org/resource/Substances',
    'de_http://de.dbpedia.org/resource/(Rayon)',
    'de_http://de.dbpedia.org/resource/Melts',
    'de_http://de.dbpedia.org/resource/Violet',
    'de_http://de.dbpedia.org/resource/Slashes',
    'de_http://de.dbpedia.org/resource/Lutheran',
    'de_http://de.dbpedia.org/resource/Car_manufacturers',
    'de_http://de.dbpedia.org/resource/Northern',
    'de_http://de.dbpedia.org/resource/Flax_land',
    'de_http://de.dbpedia.org/resource/Nurse',
    'de_http://de.dbpedia.org/resource/The_last_fight_(film)',
    'de_http://de.dbpedia.org/resource/Ecclesiastes',
    'de_http://de.dbpedia.org/resource/Sweden',
    'de_http://de.dbpedia.org/resource/Country',
    'de_http://de.dbpedia.org/resource/Goal!',
    'de_http://de.dbpedia.org/resource/Ethiopia',
    'de_http://de.dbpedia.org/resource/Of_a_kind_used_in_the_manufacture_of_agricultural_or_forestry_products',
    'de_http://de.dbpedia.org/resource/Open_Secret',
    'de_http://de.dbpedia.org/resource/Turkey',
    'de_http://de.dbpedia.org/resource/Yours',
    'de_http://de.dbpedia.org/resource/Skirts',
    'de_http://de.dbpedia.org/resource/Horsepower',
    'de_http://de.dbpedia.org/resource/Thresholds',
    'de_http://de.dbpedia.org/resource/Foreigners',
    'de_http://de.dbpedia.org/resource/Tokyo',
    'de_http://de.dbpedia.org/resource/Matching',
    'de_http://de.dbpedia.org/resource/United_Kingdom',
    'de_http://de.dbpedia.org/resource/Hats',
    'de_http://de.dbpedia.org/resource/Boards',
    'de_http://de.dbpedia.org/resource/District',
    'de_http://de.dbpedia.org/resource/The_publisher',
    'de_http://de.dbpedia.org/resource/Sea',
    'de_http://de.dbpedia.org/resource/Rings',
    'de_http://de.dbpedia.org/resource/Electropop',
    'de_http://de.dbpedia.org/resource/Army',
    'de_http://de.dbpedia.org/resource/Helmets',
    'de_http://de.dbpedia.org/resource/Rabbit',
    'de_http://de.dbpedia.org/resource/Sheepmeat',
    'de_http://de.dbpedia.org/resource/Longitudinal',
    'de_http://de.dbpedia.org/resource/Neuchâtel_(City)',
    'de_http://de.dbpedia.org/resource/Czech_Republic',
    'de_http://de.dbpedia.org/resource/Beaches',
    'de_http://de.dbpedia.org/resource/Subsidiary',
    'de_http://de.dbpedia.org/resource/Courageous',
    'de_http://de.dbpedia.org/resource/Scaffolding',
    'de_http://de.dbpedia.org/resource/Chestnuts',
    'de_http://de.dbpedia.org/resource/Flow',
    'de_http://de.dbpedia.org/resource/==References==',
    'de_http://de.dbpedia.org/resource/Sweethearts',
    'de_http://de.dbpedia.org/resource/Poultrymeat',
    'de_http://de.dbpedia.org/resource/Sheepmeat_and_goatmeat',
    'de_http://de.dbpedia.org/resource/Formalities',
    'de_http://de.dbpedia.org/resource/Substances',
    'de_http://de.dbpedia.org/resource/Cauliflowers',
    'de_http://de.dbpedia.org/resource/Netherlands',
    'de_http://de.dbpedia.org/resource/Christian_Democratic_Union',
    'de_http://de.dbpedia.org/resource/Floats',
    'de_http://de.dbpedia.org/resource/Page',
    'de_http://de.dbpedia.org/resource/New_Zealand',
    'de_http://de.dbpedia.org/resource/Maize_(corn)',
    'de_http://de.dbpedia.org/resource/District',
    'de_http://de.dbpedia.org/resource/Uses',
    'de_http://de.dbpedia.org/resource/Federal_Republic_of_Germany',
    'de_http://de.dbpedia.org/resource/Beavers',
    'de_http://de.dbpedia.org/resource/Apache_License',
    'de_http://de.dbpedia.org/resource/Issuing',
    'de_http://de.dbpedia.org/resource/Total',
    'de_http://de.dbpedia.org/resource/Apostolic_Vicariate',
    'de_http://de.dbpedia.org/resource/Turkey',
    'de_http://de.dbpedia.org/resource/Fowls_of_the_species_Gallus_domesticus',
    'de_http://de.dbpedia.org/resource/Fowls_of_the_species_Gallus_domesticus,_not_put_up_for_retail_sale',
    'de_http://de.dbpedia.org/resource/Peaches',
    'de_http://de.dbpedia.org/resource/Potatoes',
    'de_http://de.dbpedia.org/resource/Swallows',
    'de_http://de.dbpedia.org/resource/Mushrooms',
    'de_http://de.dbpedia.org/resource/Lead_guitar',
    'de_http://de.dbpedia.org/resource/Roe_deer',
    'de_http://de.dbpedia.org/resource/Rhythm_guitar',
    'de_http://de.dbpedia.org/resource/Baskets',
    'de_http://de.dbpedia.org/resource/Poultry',
    'de_http://de.dbpedia.org/resource/Loins',
    'de_http://de.dbpedia.org/resource/Seagulls',
    'de_http://de.dbpedia.org/resource/Norway',
    'de_http://de.dbpedia.org/resource/Thieves',
    'de_http://de.dbpedia.org/resource/Publisher',
    'de_http://de.dbpedia.org/resource/Wild_game',
    'de_http://de.dbpedia.org/resource/Brno',
    'de_http://de.dbpedia.org/resource/External_defenders',
    'de_http://de.dbpedia.org/resource/Rainbow_(Album)',
    'de_http://de.dbpedia.org/resource/Strasbourg',
    'de_http://de.dbpedia.org/resource/Cereals',
    'de_http://de.dbpedia.org/resource/Fowls_of_the_species_Gallus_domesticus,_not_cut_in_pieces',
    'de_http://de.dbpedia.org/resource/Communauté_de_communes_du_Val_d',
    'de_http://de.dbpedia.org/resource/Belgium',
    'de_http://de.dbpedia.org/resource/Official_Journal_of_the_European_Communities',
    'de_http://de.dbpedia.org/resource/Crime_series',
    'de_http://de.dbpedia.org/resource/Names',
    'de_http://de.dbpedia.org/resource/Cross-platform',
    'de_http://de.dbpedia.org/resource/Pendulum',
    'de_http://de.dbpedia.org/resource/Lore',
    'de_http://de.dbpedia.org/resource/Lights',
    'de_http://de.dbpedia.org/resource/Granules',
    'de_http://de.dbpedia.org/resource/Grünenbach',
    'de_http://de.dbpedia.org/resource/Chewing',
    'de_http://de.dbpedia.org/resource/Cyprus',
    'de_http://de.dbpedia.org/resource/Gourgeon',
    "de_http://de.dbpedia.org/resource/I'm_sorry,_I'm_sorry,_but_I'm_sorry.",
    'de_http://de.dbpedia.org/resource/Shards',
    'de_http://de.dbpedia.org/resource/Vilnius',
    'de_http://de.dbpedia.org/resource/Small',
    'de_http://de.dbpedia.org/resource/University_Luzern',
    'de_http://de.dbpedia.org/resource/Wolves',
    'de_http://de.dbpedia.org/resource/Small_cell',
    'de_http://de.dbpedia.org/resource/Country',
    'de_http://de.dbpedia.org/resource/Hawaiian_guitar',
    'de_http://de.dbpedia.org/resource/Wings',
    'de_http://de.dbpedia.org/resource/Lithuania',
    'de_http://de.dbpedia.org/resource/Car_manufacturers',
    'de_http://de.dbpedia.org/resource/Vessels',
    'de_http://de.dbpedia.org/resource/Sweden',
    'de_http://de.dbpedia.org/resource/Boundaries',
    'de_http://de.dbpedia.org/resource/Art_Rock'
]


def has_complex_repeats(text):
    clean_text = text.replace("-", "_").split("_")
    if len(clean_text) >= 2 and len(clean_text) - 2 > len(set(clean_text)):
        return True
    else:
        return False

# clear the error english
def error2eng(args, id_=0):
    kg_ent_id2uri_map = load_json(join(args.data_dir, args.data_name, "_".join(kgids), kgids[id_] + "_entity_txt.json"))
    ent_id2uri = dict(read_tab_lines(os.path.join(args.data_dir, args.data_name, "_".join(kgids), kgids[id_] + "_entity_id2uri.txt")))

    Temp_info = {}
    for idx in kg_ent_id2uri_map:
        uri = kg_ent_id2uri_map[idx]
        if has_complex_repeats(uri) or (uri in error_list):
            Temp_info[idx] = ent_id2uri[idx]
    dump_json(Temp_info, os.path.join(args.data_dir, args.data_name, "_".join(kgids), "Complex_" + kgids[id_] + "_entity_txt.json"))
    
    try:
        New_idmaps = load_json(os.path.join(args.data_dir, args.data_name, "_".join(kgids), "google_" + kgids[id_] + "_entity_txt.json"))
        Temp_new_info = {}
        for idx in Temp_info: 
            Temp_new_info[idx] = New_idmaps[idx]

        for idx in Temp_new_info:
            kg_ent_id2uri_map[idx] = Temp_new_info[idx]
        dump_json(kg_ent_id2uri_map, join(args.data_dir, args.data_name, "_".join(kgids), kgids[id_] + "_entity_txt.json"))
        print("ok")
    except:
        pass

# Is it an abbreviation
def is_abbreviation(phrase):
    # 移除标点符号
    phrase_cleaned = re.sub(r'[^\w\s]', '', phrase.split("/")[-1])
    
    if phrase_cleaned.isalpha() and phrase_cleaned.isupper():
        return True
    return False

# transform each abbreviation into the raw description
def abbreviation2eng(args):
    kg1_ent_id2uri_map = load_json(join(args.data_dir, args.data_name, "_".join(kgids), kgids[0] + "_entity_txt.json"))
    kg2_ent_id2uri_map = load_json(join(args.data_dir, args.data_name, "_".join(kgids), kgids[1] + "_entity_txt.json"))

    for idx in kg1_ent_id2uri_map:
        uri = kg1_ent_id2uri_map[idx]
        if is_abbreviation(uri):
            print(idx, uri)

    for idx in kg2_ent_id2uri_map:
        uri = kg2_ent_id2uri_map[idx]
        if is_abbreviation(uri):
            print(idx, uri)

if __name__ == "__main__":
    args = get_parser()
    kgids = args.kgids.split(",")
    id_ = 0

    if "en" == kgids[0]:
        id_ = 1

    error2eng(args, id_=id_)
    # abbreviation2eng(args)