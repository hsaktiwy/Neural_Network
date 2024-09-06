#include <iostream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <set>
using namespace std;

string token(double e, double d, double c)
{
    char s[49];
    union {
        double d;
        uint64_t u;
    } converter;
    converter.d = e;
    sprintf(s, "%016llx", converter.u);
    converter.d = d;
    sprintf(&s[16], "%016llx", converter.u);
    converter.d = c;
    sprintf(&s[32], "%016llx", converter.u);
    s[48] = '\0';
    return s;
}

long simpleHash(int a, int b, int sequenceSize = 256)
{
    if (a == b)
        return -1;
    long mid = (a + b);
    long result = ((mid)*(mid + (abs(a - b) + a * b))) / 2 + ((a > b) ? a + sequenceSize: b);
    return result;
}

int main()
{
    float maxpercent = 0;
    int maxpercentindex = 0;
    for (int x = 0; x < 1000; x++)
    {
        set<long> s;
        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                s.insert(simpleHash(i, j, x));
            }
        }
        float percent = (s.size()/65536.0)*100.0;
        if (percent - maxpercent > 0)
        {
            maxpercent = percent;
            maxpercentindex = x;
        }
        if (s.size() == 65536)
        {
            cout << "Test passed : " << s.size() << endl;
            cout << "Max percent : " << maxpercent << " at " << maxpercentindex << endl;
            return 0;
        }
        //el÷
        cout << x<< " : " << s.size() << " : " << percent << endl;
    }
    cout << "Max percent : " << maxpercent << " at " << maxpercentindex << endl;
    return 0;
}