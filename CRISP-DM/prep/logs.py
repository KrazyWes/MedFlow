import atexit
from dataclasses import dataclass


@dataclass
class LogSinks:
    prep_log_path: str
    du_before_log_path: str
    du_after_log_path: str

    _prep_fh: object
    _du_before_fh: object
    _du_after_fh: object

    def close(self) -> None:
        for fh in (self._prep_fh, self._du_before_fh, self._du_after_fh):
            try:
                fh.close()
            except Exception:
                pass

    def log_prep(self, message: str = "") -> None:
        self._prep_fh.write(f"{message}\n")
        self._prep_fh.flush()

    def log_du_before(self, message: str = "") -> None:
        self._du_before_fh.write(f"{message}\n")
        self._du_before_fh.flush()

    def log_du_after(self, message: str = "") -> None:
        self._du_after_fh.write(f"{message}\n")
        self._du_after_fh.flush()


def open_log_sinks(*, prep_log_path: str, du_before_log_path: str, du_after_log_path: str) -> LogSinks:
    prep_fh = open(prep_log_path, "w", encoding="utf-8")
    du_before_fh = open(du_before_log_path, "w", encoding="utf-8")
    du_after_fh = open(du_after_log_path, "w", encoding="utf-8")
    sinks = LogSinks(
        prep_log_path=prep_log_path,
        du_before_log_path=du_before_log_path,
        du_after_log_path=du_after_log_path,
        _prep_fh=prep_fh,
        _du_before_fh=du_before_fh,
        _du_after_fh=du_after_fh,
    )
    atexit.register(sinks.close)
    return sinks

