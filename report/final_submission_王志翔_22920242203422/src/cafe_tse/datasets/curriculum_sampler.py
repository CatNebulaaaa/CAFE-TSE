from __future__ import annotations


class CurriculumSchedule:
    def __init__(self, stages: list[dict]):
        self.stages = stages

    def allowed_difficulties(self, epoch: int) -> list[str]:
        for stage in self.stages:
            if epoch <= int(stage["until_epoch"]):
                return list(stage["difficulties"])
        return list(self.stages[-1]["difficulties"])

