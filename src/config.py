from pathlib import Path

ProjectRoot = Path('/') / Path('home') / Path('ray') / Path('Desktop') / Path('109-1') / Path('DataScienceComputing') / Path('FinalProject1')
DataRoot = ProjectRoot / Path('Data')
karate_dataset = DataRoot / Path('soc-karate') / Path('soc-karate.mtx')

if __name__ == '__main__':
    print(ProjectRoot)
    print(DataRoot)