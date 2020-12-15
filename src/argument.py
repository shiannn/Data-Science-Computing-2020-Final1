import argparse

def parser_vis():
    parser = argparse.ArgumentParser(description='Tabu parameter')
    parser.add_argument('-l', '--load_membership', help='load membership.npy')

    return parser.parse_args()

def parser_Tabu():
    parser = argparse.ArgumentParser(description='Tabu parameter')
    parser.add_argument('-d', '--datasets', default='karate', help='dataset used')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-m', '--max_tabu_len', default=15, type=int, help='max tabu_len of Tabu')

    parser.add_argument('-n', '--no_save', action="store_true", help='not to save')

    return parser.parse_args()

def parser_SA():
    parser = argparse.ArgumentParser(description='SA parameter')
    parser.add_argument('-d', '--datasets', default='karate', help='dataset used')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-m', '--min_temperature', default=0.5, type=float, help='min_temperature of SA')
    parser.add_argument('-e', '--epsilon', default=0.025, type=float, help='epsilon annealing')

    parser.add_argument('-n', '--no_save', action="store_true", help='not to save')

    return parser.parse_args()

def parser_ACO():
    parser = argparse.ArgumentParser(description='ACO parameter')
    parser.add_argument('-d', '--datasets', default='karate', help='dataset used')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-p', '--popu_size', default=15, type=int, help='population size of ACO')
    parser.add_argument('-e', '--evaporate', default=0.05, type=float, help='evaporate rate of ACO')
    parser.add_argument('-t', '--eta_scale', default=0.1, type=float, help='eta scale of ACO')

    parser.add_argument('-n', '--no_save', action="store_true", help='not to save')

    return parser.parse_args()

def parser_GA():
    parser = argparse.ArgumentParser(description='GA parameter')
    parser.add_argument('-d', '--datasets', default='karate', help='dataset used')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-p', '--popu_size', default=50, type=int, help='population size of GA')
    parser.add_argument('-m', '--mate_pool_size', default=30, type=int, help='mate pool size of GA')
    parser.add_argument('-k', '--k_cross_points', default=100, type=int, help='number of crosspoints of GA')
    parser.add_argument('-f', '--flip_locations', default=100, type=int, help='number of flip locations of GA')

    parser.add_argument('-n', '--no_save', action="store_true", help='not to save')

    return parser.parse_args()

def parser_PSO():
    parser = argparse.ArgumentParser(description='PSO parameter')
    parser.add_argument('-d', '--datasets', default='karate', help='dataset used')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='number of iterations')
    parser.add_argument('-p', '--popu_size', default=10, type=int, help='population size of PSO')
    parser.add_argument('-l', '--local_effect', type=float, default=1., help='local effect of PSO')
    parser.add_argument('-g', '--global_effect', type=float, default=1., help='global effect of PSO')
    parser.add_argument('-w', '--particle_weight', type=float, default=0.9, help='particle weight of PSO')

    parser.add_argument('-n', '--no_save', action="store_true", help='not to save')

    return parser.parse_args()