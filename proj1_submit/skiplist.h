#include <iostream>
#include <sstream>
#include <pthread.h>

#define BILLION  1000000000L

using namespace std;


 
template<class K,class V,int MAXLEVEL>
class skiplist_node
{
public:
    skiplist_node()
    {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            forwards[i] = NULL;
        }
    }
 
    skiplist_node(K searchKey):key(searchKey)
    {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            forwards[i] = NULL;
        }
    }
 
    skiplist_node(K searchKey,V val):key(searchKey),value(val)
    {
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            forwards[i] = NULL;
        }
    }
 
    virtual ~skiplist_node()
    {
    }
 
    K key;
    V value;
    skiplist_node<K,V,MAXLEVEL>* forwards[MAXLEVEL+1];
};
 
///////////////////////////////////////////////////////////////////////////////
 
pthread_mutex_t mm1=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mm2=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mm3=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex[17];

pthread_rwlock_t rwLock=PTHREAD_RWLOCK_INITIALIZER;
template<class K, class V, int MAXLEVEL = 16>
class skiplist
{
public:
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;
 
    skiplist(K minKey,K maxKey):m_pHeader(NULL),m_pTail(NULL),
                                max_curr_level(1),max_level(MAXLEVEL),
                                m_minKey(minKey),m_maxKey(maxKey)
    {
        m_pHeader = new NodeType(m_minKey);  // 0
        m_pTail = new NodeType(m_maxKey);   // 1000000
        for ( int i=1; i<=MAXLEVEL; i++ ) {
            m_pHeader->forwards[i] = m_pTail;
        }
        for(int i=1; i<=MAXLEVEL; i++) {
            pthread_mutex_init(&mutex[i], NULL); 
        }
    }
 
    void insert(K searchKey,V newValue)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType* currNode = m_pHeader;
       
        pthread_mutex_lock(&mm1);
        int max_curr_level_local = max_curr_level;
        pthread_mutex_unlock(&mm1);

        //pthread_rwlock_rdlock(&rwLock);
        for(int level=max_curr_level_local; level >=1; level--) {
            //pthread_mutex_lock(&mutex[level]);
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];   // shift to right
            }
            update[level] = currNode;
            //pthread_mutex_unlock(&mutex[level]);
        }
        currNode = currNode->forwards[1];   
        //pthread_rwlock_unlock(&rwLock);

        //pthread_rwlock_wrlock(&rwLock);
        if ( currNode->key == searchKey ) {
            currNode->value = newValue;
        }
        else {
            int newlevel = randomLevel();
            if ( newlevel > max_curr_level_local ) {
                for ( int level = max_curr_level_local+1; level <= newlevel; level++ ) {
                    update[level] = m_pHeader;
                }

                max_curr_level_local = newlevel;
            }
            currNode = new NodeType(searchKey,newValue);
            for ( int lv=1; lv<=max_curr_level_local; lv++ ) {
                //pthread_mutex_lock(&mutex[lv]);
                currNode->forwards[lv] = update[lv]->forwards[lv];
                update[lv]->forwards[lv] = currNode;
                //pthread_mutex_unlock(&mutex[lv]);
            }
        }

        pthread_mutex_lock(&mm1);
        max_curr_level = max_curr_level_local;
        pthread_mutex_unlock(&mm1);
        //pthread_rwlock_unlock(&rwLock);
    }
    virtual ~skiplist()
    {
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
            NodeType* tempNode = currNode;
            currNode = currNode->forwards[1];
            delete tempNode;
        }
        delete m_pHeader;
        delete m_pTail;
    }
 
 
    void erase(K searchKey)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];
            }
            update[level] = currNode;
        }
        currNode = currNode->forwards[1];
        if ( currNode->key == searchKey ) {
            for ( int lv = 1; lv <= max_curr_level; lv++ ) {
                if ( update[lv]->forwards[lv] != currNode ) {
                    break;
                }
                update[lv]->forwards[lv] = currNode->forwards[lv];
            }
            delete currNode;
            // update the max level
            while ( max_curr_level > 1 && m_pHeader->forwards[max_curr_level] == NULL ) {
                max_curr_level--;
            }
        }
    }
 
    //const NodeType* find(K searchKey)
    V find(K searchKey)
    {
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level]; // shift to right
            }
            // down the level
        }
        currNode = currNode->forwards[1];
        if ( currNode->key == searchKey ) {
            return currNode->value;
        }
        else {
            //return NULL;
            return -1;
        }
    }
 
    bool empty() const
    {
        return ( m_pHeader->forwards[1] == m_pTail );
    }
 
    std::string printList()
    {
	    int i=0;
        std::stringstream sstr;
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
            //sstr << "(" << currNode->key << "," << currNode->value << ")" << endl;
            sstr << currNode->key << " ";
            currNode = currNode->forwards[1];
	        i++;
	        if(i>200) break;
        }
        return sstr.str();
    }
 
    const int max_level;
 
protected:
    double uniformRandom()
    {
        return rand() / double(RAND_MAX);
    }
 
    int randomLevel() {
        int level = 1;
        double p = 0.5;
        while ( uniformRandom() < p && level < MAXLEVEL ) {
            level++;
        }
        return level;
    }
    K m_minKey; //4
    K m_maxKey; //4
    int max_curr_level; //4
    skiplist_node<K,V,MAXLEVEL>* m_pHeader; //8
    skiplist_node<K,V,MAXLEVEL>* m_pTail;   //8
    char padding[36];                       //36
    // 12+16+36 = 64
};
 
///////////////////////////////////////////////////////////////////////////////
 
