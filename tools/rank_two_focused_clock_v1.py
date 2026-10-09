"""First-response elapsed startup accounting, excluding blocked input waits."""
HEADERS = '''#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#if defined(__APPLE__)
#include <libproc.h>
#endif
'''

HELPER = r'''static double process_start_age_ms(){
timespec current{};
#if defined(__APPLE__)
proc_bsdinfo info{};
if(proc_pidinfo(getpid(),PROC_PIDTBSDINFO,0,&info,sizeof info)!=sizeof info||clock_gettime(CLOCK_REALTIME,&current))return -1;
const double born=double(info.pbi_start_tvsec)*1000+double(info.pbi_start_tvusec)/1000;
const double age=double(current.tv_sec)*1000+double(current.tv_nsec)/1000000-born;
#elif defined(__linux__)
char line[2048];auto file=std::fopen("/proc/self/stat","r");if(!file)return -1;
const bool read=std::fgets(line,sizeof line,file);std::fclose(file);if(!read)return -1;
auto token=std::strrchr(line,')');if(!token)return -1;token+=2;
for(int field=3;field<22;++field){token=std::strchr(token,' ');if(!token)return -1;++token;}
char*end=nullptr;const auto ticks=std::strtoull(token,&end,10);
const long rate=sysconf(_SC_CLK_TCK);
if(end==token||*end!=' '||ticks==std::numeric_limits<unsigned long long>::max()||rate<=0||clock_gettime(CLOCK_BOOTTIME,&current))return -1;
const double born=double(ticks)*1000/double(rate);
const double age=double(current.tv_sec)*1000+double(current.tv_nsec)/1000000-born;
#else
return -1;
#endif
return std::isfinite(age)&&age>=0?age:-1;
}
static double input_blocked_ms(std::chrono::steady_clock::time_point begun,std::clock_t cpu_begun){
const auto cpu_end=std::clock();if(cpu_begun<0||cpu_end<cpu_begun)return 0;
const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begun).count();
const double cpu=1000*double(cpu_end-cpu_begun)/double(CLOCKS_PER_SEC);
return std::max(0.,wall-cpu);
}
'''


def repair(source):
    source=source.replace("#include <ctime>",HEADERS+"\n#include <ctime>",1)
    anchor="#ifndef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN"
    if source.count(anchor)!=1:raise ValueError("main boundary anchor differs")
    source=source.replace(anchor,HELPER+"\n"+anchor,1)
    old="int player_id = -1;\nif (!(std::cin >> player_id)) {"
    new="int player_id = -1;\nconst auto role_wait_started=SearchClock::now();\nconst auto role_cpu_started=std::clock();\nif (!(std::cin >> player_id)) {"
    if source.count(old)!=1:raise ValueError("role input anchor differs")
    source=source.replace(old,new,1)
    old="std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\\n');"
    source=source.replace(old,old+"\ndouble startup_blocked_ms=input_blocked_ms(role_wait_started,role_cpu_started);",1)
    old="while (true) {\nstd::cin >> std::ws;"
    new="while (true) {\nconst auto request_wait_started=SearchClock::now();\nconst auto request_cpu_started=std::clock();\nstd::cin >> std::ws;"
    if source.count(old)!=1:raise ValueError("request input anchor differs")
    source=source.replace(old,new,1)
    start=source.index("if (first_execution) {\nconst std::clock_t startup_cpu")
    end=source.index("\nint opponent_move_length",start)
    source=source[:start]+'''if(first_execution){
startup_blocked_ms+=input_blocked_ms(request_wait_started,request_cpu_started);
const double age=process_start_age_ms();
if(age<0)response_deadline=response_started;
else response_deadline-=std::chrono::duration_cast<SearchClock::duration>(
std::chrono::duration<double,std::milli>(std::max(0.,age-startup_blocked_ms)));
}
'''+source[end:]
    return source
