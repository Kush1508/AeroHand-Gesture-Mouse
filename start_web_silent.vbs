Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c run_web.bat", 0, False
Set WshShell = Nothing
