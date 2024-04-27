from FeatureCloud.app.engine.app import AppState, app_state, Role

from utils import read_config, OUTPUT_DIR

from src import run

config = read_config()

@app_state('initial')
class ExecuteState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        with open(OUTPUT_DIR + "/A.csv", 'w') as out_file_1, open(OUTPUT_DIR + "/B.csv", 'w') as out_file_2:
            run(config, out_file_1, out_file_2)
        return 'terminal'
