
### defining new states
For defining new states, in general, developers can use [`AppState`](engine/README.md#appstate-defining-custom-states)
which supports further communications, transitions, logging, and operations.

#### AppState
[`AppState`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine#appstate-defining-custom-states) is the building block of FeatureCloud apps that covers
all the scenarios with the verifying mechanism. Each state of
the app should extend [`AppState`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine#appstate-defining-custom-states), which is an abstract class with two specific abstract methods:
- [`register`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-a-specific-transition-for-state-register_transition):
should be implemented by apps to register possible transitions between the current state to other states.
This method is part of verifying mechanism in FeatureCloud apps that ensures logically eligible roles can participate in the current state
and transition to other ones.
- [`run`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#executing-states-computation-run): executes all operations and calls for communication between FeatureCloud clients.
`run` is another part of the verification mechanism in the FeatureCloud library that ensures the transitions to other states are logically correct
by returning the name of the next state.

One can implement desired states in [`states.py`](states.py) and import it, which because of putting
[`app_state`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-states-to-the-app-app_state) on top of state classes,
merely importing the states and registering them into the [`app` instance](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#app-instance).

### Registering apps
For each state, developers should extend one of the abstract states and call the helper function to register automatically
the state in the default FeatureCloud app:

```angular2html
@app_state(name='initial', role=Role.BOTH, app_name='example')
class ExampleState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        self.read_config()
        self.app.log(self.config)
        return 'terminal'
```


#### App-specific files
All app-specific files should include data or codes strictly dependent on the app's functionality.

For running the app, inside a docker container, [`app.register()`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app/engine/README.md#registering-all-transitions-appregister)
should be called to register and verify all transitions; next, api and servers should mount at corresponding paths; and finally
the server is ready to run the app.

```angular2html
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)
```

All of the codes above, except for importing the app or, alternatively, implementing states, can be exactly same for all apps.

##### requirements.txt
for installing required python libraries inside the docker image, developers should provide a list of libraries in [requirements.txt](requirements.txt).
Some requirements are necessary for the FeatureCloud library, which should always be listed, are:
```angular2html
bottle
jsonpickle
joblib
numpy
bios
pydot
pyyaml
```

And the rest should be all other app-required libraries.

##### config.yml
Each app may need some hyper-parameters or arguments that the end-users should provide. Such data should be included
in [`config.yml`](https://github.com/FeatureCloud/FeatureCloud/tree/master/FeatureCloud/app#config-file-configyml), which should be read and interpreted by the app.
