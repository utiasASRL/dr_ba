class ScanLoader:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.scans = []
        self.scan_ids = []

    def add_scan(self, scan):
        self.scans.append(scan)
        self.scan_ids.append(scan.id)

    def get_scan(self, scan_id):
        return self.scans[self.scan_ids.index(scan_id)]
