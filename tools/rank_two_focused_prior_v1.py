"""Read-only ancestry validation with explicit SQLite URI handling."""
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from tools import rank_two_live_research_v2 as retained
e=retained.e
verify_stream=retained.verify_stream
inventory_header=retained.inventory_header

@contextmanager
def prior_census(frozen):
 if frozen.get('schema')!='papersoccer.rank-two.live-prior-exposure.v1' or frozen.get('historical_protected_reads') is not False:raise ValueError('invalid frozen prior exposure')
 refs=frozen['inputs']
 for ref in refs.values():verify_stream(ref)
 receipt=e.read(refs['receipt']['path']);binding=e.read(refs['binding']['path'])
 closure=e.read(refs['closure']['path']);scope=e.read(refs['scope']['path'])
 header=inventory_header(refs['inventory']['path']);database=Path(refs['database']['path'])
 if (receipt.get('status')!='complete' or receipt.get('error') is not None
     or receipt.get('historical_protected_reads') is not False
     or receipt.get('inventory')!=refs['inventory']
     or Path(receipt['checkpoint']).resolve()!=database
     or receipt.get('completed_rows')!=receipt.get('total_rows')
     or binding.get('closure')!=refs['closure'] or binding.get('scope')!=refs['scope']
     or closure.get('scope')!=refs['scope'] or closure.get('rows')!=receipt['total_rows']
     or closure.get('all_payload_inputs_hash_verified_before_census') is not True
     or scope.get('historical_protected_reads') is not False
     or header.get('binding')!=binding or header.get('status')!='complete'
     or header.get('complete_coverage') is not True
     or header.get('historical_protected_reads') is not False
     or header.get('last_fully_processed_normalized_row')!=receipt['total_rows']
     or header.get('total_normalized_rows')!=receipt['total_rows']
     or header.get('state_function_sha256')!=hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest()):
  raise ValueError('incomplete or inconsistent prior census binding')
 # A nonempty WAL is not covered by the database hash. Never checkpoint or write it.
 wal=Path(str(database)+'-wal')
 if wal.exists() and wal.stat().st_size:raise ValueError('prior census has unbound WAL changes')
 db=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
 try:
  db.execute('PRAGMA query_only=ON')
  metadata=dict(db.execute('SELECT name,value FROM metadata'))
  bootstrap=json.loads(metadata['bootstrap'])
  if json.loads(metadata['binding'])!=binding:raise ValueError('checkpoint binding mismatch')
  done=bootstrap['completed']
  for number, in db.execute('SELECT number FROM completed ORDER BY number'):
   if number!=done+1:raise ValueError('incomplete prior census checkpoint')
   done=number
  if (done!=receipt['total_rows'] or bootstrap['completed']!=receipt['bootstrap_completed_rows']
      or db.execute("SELECT count(*) FROM keys WHERE category='states'").fetchone()[0]!=receipt['states']):
   raise ValueError('incomplete prior census checkpoint')
  yield db
  verify_stream(refs['database'])
  if wal.exists() and wal.stat().st_size:raise ValueError('prior census changed during selection')
 finally:db.close()
