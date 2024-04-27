from FeatureCloud.app.engine.app import AppState, app_state, Role

from utils import read_config, OUTPUT_FILENAME

from src import run

config = read_config()

@app_state('initial')
class ExecuteState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        with open(OUTPUT_FILENAME, 'w') as out_file:
            run(config, out_file)
        return 'terminal'
