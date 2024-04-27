from FeatureCloud.app.engine.app import AppState, app_state, Role

from utils import read_config,write_output

from src import run

config = read_config()

@app_state('initial')
class ExecuteState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        run(config, write_output)
        return 'terminal'
