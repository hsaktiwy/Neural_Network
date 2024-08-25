#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <iomanip> 
#include <sstream>

using namespace std;
#define endl '\n'

long minimum(vector<long> &v, long index,long g1, long g2, long size)
{
    if (index == size)
        return (abs(g1 - g2));
    long diff1 = minimum(v, index + 1, g1 + v[index],  g2, size);
    long diff2 = minimum(v, index + 1, g1,  g2 + v[index], size);
    return (min(diff1,diff2));
}

char reverse(string&str, int a, int b, int x)
{
    if (x >=a && x<=b)
    {
        return str[(b - x) + a];
    }
    return str[x];
}

string converTostring(int hours, int min)
{
    ostringstream timeStream;
    timeStream << setw(2) << setfill('0') << hours << ":" << setw(2) << setfill('0') << min;

    return (timeStream.str());
}


int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int c,n, v;//, store = 0;
    cin >> c;
    for(int i = 0; i < c; i++)
    {
        cin >> n;
        vector<int> vec(n);
        vector<int> vis(10);
        for(int i = 0; i < n;i++)
        {
            cin >> v;
            vec[i] = v;
        }
        for(int i = 0; i < 10; i++)
        {
            vis[i] = -1;   
        }
        for(int i = 1; i < n; i++)
        {
            // cout << abs(vec[i] - vec[i - 1]) <<endl;
            vis[abs(vec[i] - vec[i - 1])]+=1;
        }
        // for(int i = 0 ;i < 10; i++)
        // {
        //     cout <<vis[i] << " ";
        // }
        // cout << endl;
        int min=10, max=-1;
        for(int i = 0; i < 10; i++)
        {
            if (vis[i] != -1)
            {
                if (vis[i] > max)
                    max = vis[i] + 1;
                if (vis[i]< min)
                    min = vis[i] + 1;
            }
        }
        cout << min << " " << max <<endl;
    }
    // cout << store << endl;
    return 0;
}

