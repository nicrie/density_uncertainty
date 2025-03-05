AVG_EARTH_RADIUS = 6371.0  # in km

SPECIES = ["G. bulloides", "N. incompta", "N. pachyderma", "G. ruber", "T. sacculifer"]
SPECIES_COMBINATIONS = {sp.split()[1]: [sp] for sp in SPECIES}
SPECIES_COMBINATIONS["all"] = [SPECIES]
SPECIES_COMBINATIONS["final"] = ["G. ruber", "N. incompta", "T. sacculifer"]

MODELS = ["1", "2", "3"]

EXPERIMENTS = {
    species: {model: "_".join([species, model]) for model in MODELS}
    for species in SPECIES_COMBINATIONS
}
