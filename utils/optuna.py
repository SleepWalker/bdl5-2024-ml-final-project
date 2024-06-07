import optuna


def optimize_with_max_trials(
    study: "optuna.study.Study",
    objective: callable,
    n_trials: int,
    states: tuple[optuna.trial.TrialState, ...] = (optuna.trial.TrialState.COMPLETE,),
    callbacks=[],
    # the rest of optuna options
    **kwargs,
):
    """
    By default the n_trials specifies trials count per worker.
    So if you use multiple processes you will have some issues:
    - you should know exactly how much workers will it be to pick correct value
    - if some of workers will reach it's n_trials faster, you'll get an idle
      worker which could do some work otherwise
    - if you'll restart the process — trial count will start from scratch without
      accounting for earlier finished trials

    Source: https://github.com/optuna/optuna/issues/1883#issuecomment-702688136
    """

    trials = study.get_trials(deepcopy=False, states=states)
    n_complete = len(trials)

    if n_complete >= n_trials:
        return

    callbacks.append(optuna.study.MaxTrialsCallback(n_trials))

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        **kwargs,
    )


def run_optuna(
    study_name: str,
    objective: callable,
    storage: str = "sqlite:///optuna_studies.db",
    n_trials=100,
    direction="minimize",
    seed: int = None,
):
    # Create a study object and optimize the objective function
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=optuna.storages.RDBStorage(
            storage,
            {
                # handle disconnections on google colab
                # https://github.com/optuna/optuna/issues/622
                "pool_pre_ping": True
            },
        ),
        load_if_exists=True,
    )
    optimize_with_max_trials(study, objective, n_trials=n_trials)

    # Best hyperparameters found
    print("Best hyperparameters: ", study.best_params)

    # Best score achieved
    print("Best score: ", study.best_value)

    return study
