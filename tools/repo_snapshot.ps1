# tworzymy foldery
New-Item -ItemType Directory -Force -Path tools, reports | Out-Null

# metadane repo
$treeFile = "reports\repo_tree.txt"
$snapFile = "reports\repo_snapshot.txt"

# "tree" nie zawsze jest na Windows – użyjemy git ls-files
(git ls-files) | ForEach-Object {
  $_ -replace '/[^/]+$','/'
} | Sort-Object -Unique | Out-File -Encoding utf8 $treeFile

# snapshot środowiska / gałęzi / zależności
$sw = New-Object System.IO.StreamWriter($snapFile, $false, [Text.UTF8Encoding]::new())
$sw.WriteLine("=== REMOTE & BRANCHES ===")
$sw.WriteLine((git remote -v | Out-String))
$sw.WriteLine((git branch -vv | Out-String))
$sw.WriteLine("=== LAST COMMITS ===")
$sw.WriteLine((git log --oneline -n 30 | Out-String))
$sw.WriteLine("=== PIP FREEZE ===")
$sw.WriteLine(((pip freeze | Sort-Object) -join "`n"))
$sw.WriteLine("=== PIPDEPTREE ===")
try { $sw.WriteLine((pipdeptree -fl | Out-String)) } catch { $sw.WriteLine("pipdeptree failed: $_") }
$sw.Close()

# analizy jakości
ruff check . 2>&1 | Tee-Object -FilePath reports\ruff.txt
mypy . --hide-error-context --pretty 2>&1 | Tee-Object -FilePath reports\mypy.txt
pytest -q --maxfail=1 --disable-warnings --cov=computations --cov-report=term-missing 2>&1 | Tee-Object -FilePath reports\pytest.txt
radon cc -s -a computations 2>&1 | Tee-Object -FilePath reports\radon_cc.txt
radon mi computations 2>&1 | Tee-Object -FilePath reports\radon_mi.txt
Write-Host "Snapshot gotowy → katalog reports\"
