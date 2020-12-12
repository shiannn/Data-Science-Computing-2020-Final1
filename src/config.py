from pathlib import Path

#ProjectRoot = Path('/') / Path('home') / Path('ray') / Path('Desktop') / Path('109-1') / Path('DataScienceComputing') / Path('FinalProject1')
ProjectRoot = Path('/') / Path('tmp2') / Path('rsh') / Path('Data-Science-Computing-2020-Final1')
DataRoot = ProjectRoot / Path('Data')
karate_dataset = DataRoot / Path('soc-karate') / Path('soc-karate.mtx')
coauthors_dataset = DataRoot / Path('ca-coauthors-dblp') / Path('ca-coauthors-dblp.mtx')

ans_dir = Path('ans_membership').mkdir(exist_ok=True)

if __name__ == '__main__':
    print(ProjectRoot)
    print(DataRoot)