import json
from queue import PriorityQueue


def queue_to_list(q):
    l = []
    while not q.empty():
        l.append(q.get()[1])
    return l


with open("source.json", "r") as file:
    projects = json.load(file)
    watchers = {}
    managers = {}

    for p in projects:
        for w in p["watchers"]:
            if w not in watchers:
                watchers[w] = PriorityQueue()
                watchers[w].put((p["priority"], p["name"]))
            else:
                watchers[w].put((p["priority"], p["name"]))

        for m in p["managers"]:
            if m not in managers:
                managers[m] = PriorityQueue()
                managers[m].put((p["priority"], p["name"]))
            else:
                managers[m].put((p["priority"], p["name"]))

    watchers = {x: queue_to_list(watchers[x]) for x in watchers}
    with open("watchers.json", "w") as w_file:
        json.dump(watchers, w_file, indent=4)

    managers = {x: queue_to_list(managers[x]) for x in managers}
    with open("managers.json", "w") as m_file:
        json.dump(managers, m_file, indent=4)
