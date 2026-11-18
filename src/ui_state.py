"""Keep evaluator responses tied to the selected server and trip."""


def select_evaluation(state, key):
    changed = state.get("evaluation_selection") != key
    if changed:
        state.pop("last_eval_result", None)
        state["evaluation_selection"] = key
    return changed


def request_evaluation(state, key, send_request):
    # Clear before sending: any network, HTTP or decoding failure leaves no old result.
    state.pop("last_eval_result", None)
    response = send_request()
    response.raise_for_status()
    result = response.json()
    if (result.get("trip_id"), result.get("partition")) != key[1:]:
        raise ValueError("The server returned results for a different trip or partition.")
    state["last_eval_result"] = result
    return result
