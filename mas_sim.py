from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("text", "image", "data")


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    load: int
    arrival: int = 0
    deadline: int = 50


@dataclass
class Message:
    tick: int
    sender: str
    recipient: str
    kind: str
    task_id: str = ""
    payload: str = ""


class MessageLog:
    def __init__(self):
        self.items: list[Message] = []

    def send(self, tick: int, sender: str, recipient: str, kind: str,
             task_id: str = "", payload: str = "") -> None:
        self.items.append(Message(tick, sender, recipient, kind, task_id, payload))

    def __len__(self) -> int:
        return len(self.items)

    def examples(self, limit: int = 12) -> list[dict[str, Any]]:
        return [vars(m) for m in self.items[:limit]]


@dataclass
class Worker:
    name: str
    speed: dict[str, float]
    queue: list[tuple[Task, int]] = field(default_factory=list)
    current: tuple[Task, int, int] | None = None
    pending_load: int = 0
    alive: bool = True

    def duration(self, task: Task) -> int:
        return max(1, math.ceil(task.load / self.speed[task.kind]))

    def workload(self) -> int:
        current_left = self.current[2] if self.current else 0
        return current_left + sum(self.duration(task) for task, _ in self.queue) + self.pending_load

    def estimate(self, task: Task) -> int:
        return self.workload() + self.duration(task)

    def enqueue(self, task: Task, lease: int = 0) -> None:
        if self.alive:
            self.queue.append((task, lease))

    def tick(self) -> tuple[Task, int] | None:
        if not self.alive:
            return None
        if self.current is None and self.queue:
            task, lease = self.queue.pop(0)
            self.current = (task, lease, self.duration(task))
        if self.current is None:
            return None
        task, lease, left = self.current
        left -= 1
        if left <= 0:
            self.current = None
            return task, lease
        self.current = (task, lease, left)
        return None

    def cancel(self, task_id: str) -> None:
        self.queue = [(task, lease) for task, lease in self.queue if task.id != task_id]
        if self.current and self.current[0].id == task_id:
            self.current = None


def configs() -> list[tuple[str, dict[str, float]]]:
    return [
        ("A1", {"text": 5.5, "image": 1.6, "data": 2.2}),
        ("A2", {"text": 2.0, "image": 5.1, "data": 2.0}),
        ("A3", {"text": 2.5, "image": 2.1, "data": 5.0}),
        ("A4", {"text": 3.5, "image": 3.2, "data": 3.3}),
    ]


def make_tasks(seed: int, count: int = 48, arrival_span: int = 35) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        kind = rng.choices(KINDS, weights=[4, 3, 3])[0]
        arr = rng.randrange(arrival_span + 1) if arrival_span else 0
        tasks.append(Task(f"T{i+1:03}", kind, rng.randint(8, 27), arr, arr + 22))
    return sorted(tasks, key=lambda x: (x.arrival, x.id))


def allocation_trial(seed: int = 0, mode: str = "central") -> tuple[dict[str, float], MessageLog]:
    """Lab 2: central = фиксированная минимальная длительность;
    cnp = FIPA-like CFP/PROPOSE/ACCEPT/REJECT, 1 такт на доставку решения.
    """
    if mode not in {"central", "cnp"}:
        raise ValueError(mode)
    workers = {name: Worker(name, speeds) for name, speeds in configs()}
    tasks = make_tasks(seed)
    log = MessageLog()
    released = 0
    completed: dict[str, int] = {}
    pending: list[tuple[int, str, Task]] = []
    for t in range(500):
        ready = [rec for rec in pending if rec[0] <= t]
        pending = [rec for rec in pending if rec[0] > t]
        for _, wid, task in ready:
            workers[wid].pending_load -= workers[wid].duration(task)
            workers[wid].enqueue(task)
        while released < len(tasks) and tasks[released].arrival <= t:
            task = tasks[released]
            released += 1
            if mode == "central":
                # Центр знает профили производительности, но НЕ учитывает очереди.
                winner = min(workers.values(), key=lambda w: (w.duration(task), w.name))
                log.send(t, "D", winner.name, "ASSIGN", task.id)
                winner.enqueue(task)
            else:
                bids = []
                for w in workers.values():
                    log.send(t, "D", w.name, "CFP", task.id)
                    bid = w.estimate(task)
                    log.send(t, w.name, "D", "PROPOSE", task.id, f"eta={bid}")
                    bids.append((bid, w.name))
                winner_name = min(bids)[1]
                for w in workers.values():
                    kind = "ACCEPT_PROPOSAL" if w.name == winner_name else "REJECT_PROPOSAL"
                    log.send(t, "D", w.name, kind, task.id)
                # Резервирование результата переговоров устраняет двойное назначение.
                selected = workers[winner_name]
                selected.pending_load += selected.duration(task)
                pending.append((t + 1, winner_name, task))
        for w in workers.values():
            result = w.tick()
            if result:
                task, _ = result
                completed[task.id] = t + 1
                log.send(t + 1, w.name, "D", "INFORM_DONE", task.id)
        if len(completed) == len(tasks):
            break
    assert len(completed) == len(tasks), "Симуляция не завершилась"
    latencies = [completed[task.id] - task.arrival for task in tasks]
    late_count = sum(completed[task.id] > task.deadline for task in tasks)
    result = {
        "seed": seed,
        "method": mode,
        "tasks": len(tasks),
        "makespan": max(completed.values()) - min(x.arrival for x in tasks),
        "mean_flow_time": statistics.mean(latencies),
        "max_flow_time": max(latencies),
        "late_tasks": late_count,
        "late_pct": 100 * late_count / len(tasks),
        "messages": len(log),
        "coordination_cost": len(log) * 0.05,
    }
    return result, log


def jain_fairness(values: list[float]) -> float:
    """Индекс Джейна по долям выполненного спроса."""
    d = len(values) * sum(x*x for x in values)
    return (sum(values) ** 2) / d if d else 1.0


def resource_trial(seed: int = 0, mode: str = "naive") -> tuple[dict[str, float], MessageLog]:
    """Lab 3: четыре агента конкурируют за один GPU-слот на каждый такт.

    naive: независимые попытки, при коллизии никто не исполняет задачу;
    negotiation: CFP->BID->OFFER->ACCEPT/COUNTER->GRANT, возраст приоритета.
    """
    if mode not in {"naive", "negotiation"}:
        raise ValueError(mode)
    rng = random.Random(seed)
    demand = [8, 10, 12, 7]
    remaining = demand.copy()
    done = [0, 0, 0, 0]
    ages = [0, 0, 0, 0]
    log = MessageLog()
    conflict_slots = 0
    occupied_slots = 0
    horizon = 30  # меньше суммарного спроса: ограничение ресурса реально действует
    for t in range(horizon):
        contenders = [i for i in range(4) if remaining[i] > 0]
        if not contenders:
            break
        if mode == "naive":
            # Независимые агенты не согласовывают намерения.
            requested = [i for i in contenders if rng.random() < [0.43, 0.48, 0.45, 0.4][i]]
            for i in requested:
                log.send(t, f"A{i+1}", "GPU", "REQUEST_SLOT", payload=f"slot={t}")
            if len(requested) >= 2:
                conflict_slots += 1
            elif len(requested) == 1:
                i = requested[0]
                done[i] += 1
                remaining[i] -= 1
                occupied_slots += 1
                log.send(t, "GPU", f"A{i+1}", "SLOT_USED")
        else:
            bids: list[tuple[float, int, float]] = []
            for i in contenders:
                log.send(t, "GPU", f"A{i+1}", "CFP_SLOT")
                # Локальная ставка: частная срочность плюс возраст ожидания.
                private_value = 4.1 + 0.10*remaining[i] + 0.63*ages[i] + rng.random()*0.9
                ratio = done[i] / demand[i]
                priority = private_value + 2.2*(1-ratio)
                log.send(t, f"A{i+1}", "GPU", "BID", payload=f"value={private_value:.2f}")
                bids.append((priority, i, private_value))
            bids.sort(key=lambda item: (-item[0], item[1]))
            winner = None
            for _, i, value in bids:
                asked = 7.5 + 0.1*len(contenders)
                log.send(t, "GPU", f"A{i+1}", "OFFER", payload=f"price={asked:.2f}")
                if value >= asked:
                    log.send(t, f"A{i+1}", "GPU", "ACCEPT", payload=f"price={asked:.2f}")
                    winner = i
                    break
                # Ответное предложение: агент раскрывает максимум, который готов платить.
                log.send(t, f"A{i+1}", "GPU", "COUNTER", payload=f"price={value:.2f}")
                if value >= 4.0:  # резервная цена владельца ресурса
                    log.send(t, "GPU", f"A{i+1}", "ACCEPT_COUNTER", payload=f"price={value:.2f}")
                    winner = i
                    break
                log.send(t, "GPU", f"A{i+1}", "REJECT")
            if winner is not None:
                log.send(t, "GPU", f"A{winner+1}", "GRANT", payload=f"slot={t}")
                done[winner] += 1
                remaining[winner] -= 1
                occupied_slots += 1
                ages[winner] = 0
            for i in contenders:
                if i != winner:
                    ages[i] += 1
                    log.send(t, "GPU", f"A{i+1}", "DEFER", payload=f"next_slot={t+1}")
    ratios = [done[i] / demand[i] for i in range(4)]
    result = {
        "seed": seed,
        "method": mode,
        "demand": sum(demand),
        "completed": sum(done),
        "throughput_pct": 100*sum(done)/sum(demand),
        "conflict_slots": conflict_slots,
        "occupied_slots": occupied_slots,
        "utilization_pct": 100*occupied_slots/horizon,
        "jain_fairness": jain_fairness(ratios),
        "messages": len(log),
        "coordination_cost": len(log) * 0.05,
    }
    return result, log


def reliability_trial(seed: int = 0, scenario: str = "message_loss", protected: bool = False) -> tuple[dict[str, float], MessageLog]:
    """Lab 4: злонамеренные и аварийные воздействия на тех же исполнителей и диспетчер.

    Защита: ACK+повторная отправка, heartbeat/переназначение,
    проверка неотрицательных ставок, привязка INFORM к отправителю и lease ID.
    """
    if scenario not in {"message_loss", "worker_crash", "tampered_bid", "forged_completion"}:
        raise ValueError(scenario)
    rng = random.Random(seed + 1000)
    workers = {name: Worker(name, speeds) for name, speeds in configs()}
    tasks = make_tasks(seed, count=28, arrival_span=0)
    by_id = {task.id: task for task in tasks}
    states = {task.id: "new" for task in tasks}
    who: dict[str, str] = {}
    lease_no = {task.id: 0 for task in tasks}
    sent_at: dict[str, int] = {}
    acked: set[str] = set()
    genuine_complete: dict[str, int] = {}
    log = MessageLog()
    retries = 0
    rejected = 0
    false_accepted = 0

    def dispatch(task: Task, t: int) -> None:
        nonlocal rejected
        candidates: list[tuple[int, str]] = []
        for w in workers.values():
            if not w.alive:
                continue
            reported = -100 if scenario == "tampered_bid" and w.name == "A3" else w.estimate(task)
            log.send(t, "D", w.name, "CFP", task.id)
            log.send(t, w.name, "D", "PROPOSE", task.id, f"eta={reported}")
            if protected and reported < 1:
                rejected += 1
                log.send(t, "D", w.name, "REJECT_INVALID_BID", task.id)
                continue
            candidates.append((reported, w.name))
        if not candidates:
            raise RuntimeError("Нет работоспособных агентов для назначения")
        wid = min(candidates)[1]
        lease_no[task.id] += 1
        lease = lease_no[task.id]
        states[task.id] = "assigned"
        who[task.id] = wid
        sent_at[task.id] = t
        acked.discard(task.id)
        log.send(t, "D", wid, "ASSIGN", task.id, f"lease={lease}")
        dropped = scenario == "message_loss" and rng.random() < 0.45
        if dropped:
            log.send(t, "NETWORK", "D", "DROP_NOTICE", task.id)
        else:
            workers[wid].enqueue(task, lease)
            acked.add(task.id)
            log.send(t, wid, "D", "ACK", task.id, f"lease={lease}")

    for task in tasks:
        dispatch(task, 0)
    # Подделываются именно заявки, находящиеся в исполнении, а не ещё не выданные.
    forged_ids = [task.id for task in tasks[:6]]
    for t in range(55):
        if scenario == "worker_crash" and t == 5:
            workers["A2"].alive = False
            log.send(t, "A2", "D", "HEARTBEAT_LOST")
        if scenario == "forged_completion" and t == 1:
            for tid in forged_ids:
                if states[tid] != "assigned":
                    continue
                log.send(t, "INTRUDER", "D", "INFORM_DONE", tid, "lease=0")
                if protected:
                    rejected += 1
                    log.send(t, "D", "MONITOR", "REJECT_FORGED", tid)
                else:
                    # Ошибочный статус завершения вызывает отмену настоящей работы.
                    states[tid] = "claimed_done"
                    workers[who[tid]].cancel(tid)
                    false_accepted += 1
        if protected:
            # Watchdog: необработанная отправка без ACK через два такта.
            for tid in list(who):
                if states[tid] != "assigned":
                    continue
                if tid not in acked and t - sent_at[tid] >= 2:
                    retries += 1
                    log.send(t, "MONITOR", "D", "RETRY_TIMEOUT", tid)
                    dispatch(by_id[tid], t)
            # Heartbeat-реакция: после двух пропущенных тактов переназначить
            # все незавершенные задачи отказавшего агента.
            if scenario == "worker_crash" and t == 7:
                for tid, wid in list(who.items()):
                    if wid == "A2" and states[tid] == "assigned":
                        retries += 1
                        log.send(t, "MONITOR", "D", "FAILOVER", tid)
                        dispatch(by_id[tid], t)
        for worker in workers.values():
            result = worker.tick()
            if not result:
                continue
            task, completed_lease = result
            log.send(t + 1, worker.name, "D", "INFORM_DONE", task.id,
                     f"lease={completed_lease}")
            # stale completion: после повторного назначения старый lease
            # не может ошибочно закрыть задачу.
            if protected and (who.get(task.id) != worker.name or lease_no[task.id] != completed_lease):
                rejected += 1
                log.send(t + 1, "D", worker.name, "REJECT_STALE", task.id)
                continue
            if states.get(task.id) == "assigned":
                states[task.id] = "done"
                genuine_complete[task.id] = t+1
        if len(genuine_complete) == len(tasks):
            break
    true_completed = len(genuine_complete)
    result = {
        "seed": seed,
        "scenario": scenario,
        "protection": "on" if protected else "off",
        "tasks": len(tasks),
        "true_completed": true_completed,
        "true_completion_pct": 100*true_completed/len(tasks),
        "unfinished_or_false": len(tasks)-true_completed,
        "false_accepted": false_accepted,
        "rejected_invalid": rejected,
        "retries": retries,
        "messages": len(log),
        "mean_complete_time": statistics.mean(genuine_complete.values()) if genuine_complete else 55.0,
    }
    return result, log


def aggregate(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[x] for x in group_cols), []).append(row)
    out = []
    for key, group in groups.items():
        record = {col: value for col, value in zip(group_cols, key)}
        record["trials"] = len(group)
        for col in group[0]:
            if col in group_cols or col == "seed":
                continue
            if isinstance(group[0][col], (int, float)):
                record[col] = round(statistics.mean(x[col] for x in group), 3)
        out.append(record)
    return out


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_all(outdir: Path, trials: int = 20) -> dict[str, list[dict[str, Any]]]:
    if trials < 1:
        raise ValueError("trials >= 1")
    outdir.mkdir(parents=True, exist_ok=True)
    lab2 = [allocation_trial(seed, method)[0] for seed in range(trials)
            for method in ("central", "cnp")]
    lab3 = [resource_trial(seed, method)[0] for seed in range(trials)
            for method in ("naive", "negotiation")]
    lab4 = [reliability_trial(seed, scenario, protected)[0]
            for seed in range(trials)
            for scenario in ("message_loss", "worker_crash", "tampered_bid", "forged_completion")
            for protected in (False, True)]
    summary = {
        "lab2": aggregate(lab2, ["method"]),
        "lab3": aggregate(lab3, ["method"]),
        "lab4": aggregate(lab4, ["scenario", "protection"]),
    }
    for name, values in (("lab2", lab2), ("lab3", lab3), ("lab4", lab4)):
        write_csv(outdir / f"{name}_trials.csv", values)
        write_csv(outdir / f"{name}_summary.csv", summary[name])
    traces = {
        "lab2_central": allocation_trial(0, "central")[1].examples(30),
        "lab2_cnp": allocation_trial(0, "cnp")[1].examples(30),
        "lab3_naive": resource_trial(0, "naive")[1].examples(30),
        "lab3_negotiation": resource_trial(0, "negotiation")[1].examples(30),
    }
    for scenario in ("message_loss", "worker_crash", "tampered_bid", "forged_completion"):
        traces[f"lab4_{scenario}_protected"] = reliability_trial(0, scenario, True)[1].examples(5000)
    (outdir / "message_traces.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Демонстрация МАС: 4 лабораторные работы")
    parser.add_argument("--lab", choices=["all", "2", "3", "4"], default="all")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--trace", action="store_true", help="Вывести трассу обмена сообщений")
    args = parser.parse_args()
    if args.lab == "all":
        summary = run_all(args.out, args.trials)
    elif args.lab == "2":
        r = [allocation_trial(s, m)[0] for s in range(args.trials) for m in ("central", "cnp")]
        summary = {"lab2": aggregate(r, ["method"])}
        write_csv(args.out / "lab2_trials.csv", r)
    elif args.lab == "3":
        r = [resource_trial(s, m)[0] for s in range(args.trials) for m in ("naive", "negotiation")]
        summary = {"lab3": aggregate(r, ["method"])}
        write_csv(args.out / "lab3_trials.csv", r)
    else:
        r = [reliability_trial(s, scene, p)[0] for s in range(args.trials)
             for scene in ("message_loss", "worker_crash", "tampered_bid", "forged_completion") for p in (False, True)]
        summary = {"lab4": aggregate(r, ["scenario", "protection"])}
        write_csv(args.out / "lab4_trials.csv", r)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.trace:
        _, log = allocation_trial(0, "cnp")
        print("\nПример трассы CNP:\n", json.dumps(log.examples(16), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
