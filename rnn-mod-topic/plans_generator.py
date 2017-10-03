#!/usr/bin/env python2
#-*- coding: utf-8 -*-

"""
Generate sentence plans for sampler.lua

Sentence plan is a combined topic/event vector representation:

    I	10	1 1 173 1 174 7
    decided	10	1 1 173 0 174 7
    one	10	1 1 173 0 174 6
    day	10	1 1 173 0 174 5
    to	10	1 1 173 0 174 4

For aritficial senence plan we do not need any words:

    - 10  1 1 173 1 174 7
    - 10  1 1 173 0 174 7
    - 10 1 1 173 0 174 6

"""

import sys
import random


TOPICS = ["bath", "bicycle", "bus", "cake", "flight", "grocery",
          "haircut", "library", "train", "tree"]

TOPICS2ID = {
    "bath": 1,
    "bicycle": 2,
    "bus": 3,
    "cake": 4,
    "flight": 5,
    "grocery": 6,
    "haircut": 7,
    "library": 8,
    "train": 9,
    "tree": 10
    }


EVENT2ID = {
    "Evoking": 1,
    "RelNScrEv": 2,
    "ScrEv_add_ingredients": 3,
    "ScrEv_apply_soap": 4,
    "ScrEv_arrive_destination": 5,
    "ScrEv_ask_librarian": 6,
    "ScrEv_board_bus": 7,
    "ScrEv_board_plane": 8,
    "ScrEv_bring_vehicle": 9,
    "ScrEv_browse_releases": 10,
    "ScrEv_brush_hair": 11,
    "ScrEv_buckle_seat_belt": 12,
    "ScrEv_bus_comes": 13,
    "ScrEv_bus_leaves": 14,
    "ScrEv_bus_stops": 15,
    "ScrEv_cashier_scan/weight": 16,
    "ScrEv_change_bus": 17,
    "ScrEv_check": 18,
    "ScrEv_check_catalog": 19,
    "ScrEv_check_in": 20,
    "ScrEv_check_list": 21,
    "ScrEv_check_luggage": 22,
    "ScrEv_check_new_tire": 23,
    "ScrEv_check_off": 24,
    "ScrEv_check_temp": 25,
    "ScrEv_check_time-table": 26,
    "ScrEv_choose_recipe": 27,
    "ScrEv_choose_tree": 28,
    "ScrEv_close_drain": 29,
    "ScrEv_comb": 30,
    "ScrEv_conductor_checks": 31,
    "ScrEv_cool_down": 32,
    "ScrEv_customer_opinion": 33,
    "ScrEv_cut": 34,
    "ScrEv_decorate": 35,
    "ScrEv_dig_hole": 36,
    "ScrEv_door_opens": 37,
    "ScrEv_dry": 38,
    "ScrEv_eat": 39,
    "ScrEv_enter": 40,
    "ScrEv_enter_bathroom": 41,
    "ScrEv_examine_tire": 42,
    "ScrEv_exit_plane": 43,
    "ScrEv_fill_water/wait": 44,
    "ScrEv_find_bus": 45,
    "ScrEv_find_place": 46,
    "ScrEv_find_terminal": 47,
    "ScrEv_free_wheel": 48,
    "ScrEv_get_airport": 49,
    "ScrEv_get_bus_stop": 50,
    "ScrEv_get_called": 51,
    "ScrEv_get_destination_airport": 52,
    "ScrEv_get_dressed": 53,
    "ScrEv_get_groceries": 54,
    "ScrEv_get_ingredients": 55,
    "ScrEv_get_library": 56,
    "ScrEv_get_off": 57,
    "ScrEv_get_on": 58,
    "ScrEv_get_out_bath": 59,
    "ScrEv_get_plane": 60,
    "ScrEv_get_platform": 61,
    "ScrEv_get_receipt": 62,
    "ScrEv_get_salon": 63,
    "ScrEv_get_shelf": 64,
    "ScrEv_get_ticket": 65,
    "ScrEv_get_tickets": 66,
    "ScrEv_get_tire": 67,
    "ScrEv_get_tools": 68,
    "ScrEv_get_towel": 69,
    "ScrEv_get_train_station": 70,
    "ScrEv_get_tree": 71,
    "ScrEv_get_utensils": 72,
    "ScrEv_go_check_in": 73,
    "ScrEv_go_check_out": 74,
    "ScrEv_go_checkout": 75,
    "ScrEv_go_exit": 76,
    "ScrEv_go_grocery": 77,
    "ScrEv_go_security_checks": 78,
    "ScrEv_go_target_location": 79,
    "ScrEv_grease_cake_tin": 80,
    "ScrEv_landing": 81,
    "ScrEv_lay_bike_down": 82,
    "ScrEv_leave": 83,
    "ScrEv_leave_airport": 84,
    "ScrEv_leave_tip": 85,
    "ScrEv_listen_call": 86,
    "ScrEv_listen_crew": 87,
    "ScrEv_look_for_book": 88,
    "ScrEv_look_mirror": 89,
    "ScrEv_loose_nut": 90,
    "ScrEv_make_appointment": 91,
    "ScrEv_make_dough": 92,
    "ScrEv_make_hair_style": 93,
    "ScrEv_make_list": 94,
    "ScrEv_meet_people": 95,
    "ScrEv_move_exit": 96,
    "ScrEv_move_in_salon": 97,
    "ScrEv_move_section": 98,
    "ScrEv_note_shelf": 99,
    "ScrEv_notice_problem": 100,
    "ScrEv_obtain_card": 101,
    "ScrEv_open_drain": 102,
    "ScrEv_other": 103,
    "ScrEv_pack_groceries": 104,
    "ScrEv_pack_luggage": 105,
    "ScrEv_pay": 106,
    "ScrEv_place_fertilizers": 107,
    "ScrEv_place_root": 108,
    "ScrEv_pour_dough": 109,
    "ScrEv_preheat": 110,
    "ScrEv_prepare_bath": 111,
    "ScrEv_prepare_ingredients": 112,
    "ScrEv_prepare_wash": 113,
    "ScrEv_present_ID/ticket": 114,
    "ScrEv_present_boarding_pass": 115,
    "ScrEv_press_stop_button": 116,
    "ScrEv_pull_air_pin": 117,
    "ScrEv_put_bubble_bath_scent": 118,
    "ScrEv_put_cake_oven": 119,
    "ScrEv_put_conveyor": 120,
    "ScrEv_put_new_tire": 121,
    "ScrEv_put_on_cape": 122,
    "ScrEv_put_patch/seal": 123,
    "ScrEv_put_towel": 124,
    "ScrEv_refill_hole": 125,
    "ScrEv_refill_tire_air": 126,
    "ScrEv_register": 127,
    "ScrEv_relax": 128,
    "ScrEv_remove_cap": 129,
    "ScrEv_remove_tire": 130,
    "ScrEv_retrieve_luggage": 131,
    "ScrEv_return_book": 132,
    "ScrEv_return_shop_cart": 133,
    "ScrEv_ride": 134,
    "ScrEv_ride_bike": 135,
    "ScrEv_scalp_massage": 136,
    "ScrEv_set_time": 137,
    "ScrEv_show_card": 138,
    "ScrEv_sink_water": 139,
    "ScrEv_sit_down": 140,
    "ScrEv_small_talk": 141,
    "ScrEv_spend_time_bus": 142,
    "ScrEv_spend_time_flight": 143,
    "ScrEv_spend_time_train": 144,
    "ScrEv_stow_away_luggage": 145,
    "ScrEv_take_bags": 146,
    "ScrEv_take_book": 147,
    "ScrEv_take_clean_clothes": 148,
    "ScrEv_take_home": 149,
    "ScrEv_take_off": 150,
    "ScrEv_take_off_preparations": 151,
    "ScrEv_take_out_cake_tin": 152,
    "ScrEv_take_out_oven": 153,
    "ScrEv_take_seat": 154,
    "ScrEv_take_shop_cart": 155,
    "ScrEv_take_tire_off": 156,
    "ScrEv_talk_haircut": 157,
    "ScrEv_tamp_dirt": 158,
    "ScrEv_tie_stakes_up": 159,
    "ScrEv_train_arrives": 160,
    "ScrEv_train_departs": 161,
    "ScrEv_turn_off_oven": 162,
    "ScrEv_turn_water_off": 163,
    "ScrEv_turn_water_on": 164,
    "ScrEv_undress": 165,
    "ScrEv_unwrap_root": 166,
    "ScrEv_use_computer": 167,
    "ScrEv_wait": 168,
    "ScrEv_wait_boarding": 169,
    "ScrEv_wash": 170,
    "ScrEv_water": 171,
    "Unclear": 172,
    "UnrelEv": 173
}


# generate sentence plan using random choice
def random_event():
    # get event type and id
    event, eid = random.choice(TOPICS2ID.items())
    # set event occurrence per event type
    enum = random.choice([1, 2])
    return eid, enum


# generate random plan
def gen_rnd_plan():
    # set the number of event types present in a plan
    ecnt = random.choice([0, 1, 2])
    rnd_events = [random_event() for i in range(ecnt)]
    # convert given topic to id
    plan = '- {0}'.format(TOPICS2ID[sys.argv[2]])
    for event in rnd_events:
        plan += ' {0} {1}'.format(*event)
    return plan


assert(sys.argv[1]), 'Please provide the number of sentence plans to generate!'
assert(len(sys.argv) > 2), 'Please provide a valid topic {0}'.format(str(TOPICS2ID))

# write plans to file
file = open('plans.txt', 'a')
for i in range(int(sys.argv[1])):
    file.write(gen_rnd_plan())
    file.write('\n')
file.close()
