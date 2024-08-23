import json
import sympy

def get_vteam_params(model):
    json_vteam_params = """ 
    {
        "Yalon2012" : {
        "description" : "HfO2 Yalon 2012",
        "alpha_off" : "1.0",
        "alpha_on" : "3.0",
        "v_off" : "0.5",
        "v_on" : "-0.53",
        "r_off" : "2.5 * (10 ** 3)",
        "r_on" : "100.0",
        "k_off" : "40.30 * (10 ** -9)",
        "k_on" : "-80.0",
        "d" : "(10 * 10 ** -9)"
        }, 
        "Ho2017" : {
        "description": "HfO2 Ho 2017",
        "alpha_off" : "2.0",
        "alpha_on" : "1.0",
        "v_off" : "0.7",
        "v_on" : "-0.45",
        "r_off" : "173.8 * (10 ** 3)",
        "r_on" : "7000.0",
        "k_off" : "28.92 * (10 ** -9)",
        "k_on" : "-198.72 * (10 ** -3)",
        "d" : "(10 * 10 ** -9)"
        },
        "Campbell2017" : {
        "description": "Ag/SnSe/Ge2Se3 Campbell 2017",
        "alpha_off" : "1.0",
        "alpha_on" : "1.0",
        "v_off" : "90 * (10 ** -3)",
        "v_on" : "-150 * (10 ** -3)",
        "r_off" : "150 * (10 ** 3)",
        "r_on" : "1400.0",
        "k_off" : "11.27",
        "k_on" : "-5.74 * (10 ** -3)",
        "d" : "((1/1000) * 10 ** -9)"
        },
        "Oblea2010" : {
        "description" : "Ag/Ag2Se/Ge2Se3 Oblea 2010",
        "alpha_off" : "3.0",
        "alpha_on" : "3.0",
        "v_off" : "160 * (10 ** -3)",
        "v_on" : "-150 * (10 ** -3)",
        "r_off" : "1070.0",
        "r_on" : "387.0",
        "k_off" : "2.49 * (10 ** -6)",
        "k_on" : "-220 * (10 ** -6)",
        "d" : "(10 * 10 ** -9)"
        },
        "Linear12" : {
        "description" : "Linear model, operating voltage 1.2",
        "alpha_off" : "1.0",
        "alpha_on" : "1.0",
        "v_off" : "0.6",
        "v_on" : "-0.6",
        "r_off" : "10000.0",
        "r_on" : "1000.0",
        "k_off" : "1.0",
        "k_on" : "-1.0",
        "d" : "0.5"
        },
        "template" : {
        "description" : "",
        "alpha_off" : "",
        "alpha_on" : "",
        "v_off" : "",
        "v_on" : "",
        "r_off" : "",
        "r_on" : "",
        "k_off" : "",
        "k_on" : "",
        "d" : ""
        } 
    }
    """

    # parse x:
    vteam_params = json.loads(json_vteam_params)
    selected_params = vteam_params[model]
    for key in selected_params:
        if key != "description":
            selected_params[key] = float(sympy.sympify(selected_params[key]))
        print(f"{key}:{selected_params[key]}")
    print(f"\n")

    return selected_params
