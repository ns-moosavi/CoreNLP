//
// StanfordCoreNLP -- a suite of NLP tools
// Copyright (c) 2009-2010 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//

package edu.stanford.nlp.dcoref;

import java.io.Serializable;
import java.util.*;
import java.util.logging.Logger;

import edu.stanford.nlp.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.dcoref.Dictionaries.Gender;
import edu.stanford.nlp.dcoref.Dictionaries.Number;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.Generics;

/**
 * One cluster for the SieveCoreferenceSystem.
 *
 * @author Heeyoung Lee
 */
public class CorefCluster implements Serializable{

  private static final long serialVersionUID = 8655265337578515592L;

  protected final Set<Mention> corefMentions;
  public final int clusterID;

  // Attributes for cluster - can include multiple attribute e.g., {singular, plural}
  protected final Set<Number> numbers;
  protected final Set<Gender> genders;
  protected final Set<Animacy> animacies;
  protected final Set<String> nerStrings;
  protected final Set<String> heads;

  /** All words in this cluster - for word inclusion feature  */
  public final Set<String> words;

  /** The first mention in this cluster */
  protected Mention firstMention;

  /** Return the most representative mention in the chain.
   *  A proper noun mention or a mention with more pre-modifiers is preferred.
   */
  protected Mention representative;

  public int getClusterID(){ return clusterID; }
  public Set<Mention> getCorefMentions() { return corefMentions; }
  public Mention getFirstMention() { return firstMention; }
  public Mention getRepresentativeMention() { return representative; }

  public CorefCluster(int ID) {
    clusterID = ID;
    corefMentions = Generics.newHashSet();
    numbers = EnumSet.noneOf(Number.class);
    genders = EnumSet.noneOf(Gender.class);
    animacies = EnumSet.noneOf(Animacy.class);
    nerStrings = Generics.newHashSet();
    heads = Generics.newHashSet();
    words = Generics.newHashSet();
    firstMention = null;
    representative = null;
  }

  public CorefCluster(int ID, Set<Mention> mentions){
    this(ID);
    // Register mentions
    corefMentions.addAll(mentions);
    // Get list of mentions in textual order
    List<Mention> sortedMentions = new ArrayList<>(mentions.size());
    sortedMentions.addAll(mentions);
    Collections.sort(sortedMentions, new CorefChain.MentionComparator());
    // Set default for first / representative mention
    if (sortedMentions.size() > 0) {
      firstMention = sortedMentions.get(0);
      representative = sortedMentions.get(0); // will be updated below
    }

    for (Mention m : sortedMentions) {
      // Add various information about mentions to cluster
      animacies.add(m.animacy);
      genders.add(m.gender);
      numbers.add(m.number);
      nerStrings.add(m.nerString);
      heads.add(m.headString);
      if(!m.isPronominal()){
        for(CoreLabel w : m.originalSpan){
          words.add(w.get(CoreAnnotations.TextAnnotation.class).toLowerCase());
        }
      }
      // Update representative mention, if appropriate
      if (m != representative && m.moreRepresentativeThan(representative)) {
        assert !representative.moreRepresentativeThan(m);
        representative = m;
      }
    }
  }

  /** merge 2 clusters: to = to + from */
  public static void mergeClusters(CorefCluster to, CorefCluster from) {
    int toID = to.clusterID;
    for (Mention m : from.corefMentions){
      m.corefClusterID = toID;
    }
    if(Constants.SHARE_ATTRIBUTES){
      to.numbers.addAll(from.numbers);
      if(to.numbers.size() > 1 && to.numbers.contains(Number.UNKNOWN)) {
        to.numbers.remove(Number.UNKNOWN);
      }

      to.genders.addAll(from.genders);
      if(to.genders.size() > 1 && to.genders.contains(Gender.UNKNOWN)) {
        to.genders.remove(Gender.UNKNOWN);
      }

      to.animacies.addAll(from.animacies);
      if(to.animacies.size() > 1 && to.animacies.contains(Animacy.UNKNOWN)) {
        to.animacies.remove(Animacy.UNKNOWN);
      }

      to.nerStrings.addAll(from.nerStrings);
      if(to.nerStrings.size() > 1 && to.nerStrings.contains("O")) {
        to.nerStrings.remove("O");
      }
      if(to.nerStrings.size() > 1 && to.nerStrings.contains("MISC")) {
        to.nerStrings.remove("MISC");
      }
    }

    to.heads.addAll(from.heads);
    to.corefMentions.addAll(from.corefMentions);
    to.words.addAll(from.words);
    if(from.firstMention.appearEarlierThan(to.firstMention) && !from.firstMention.isPronominal()) {
      assert !to.firstMention.appearEarlierThan(from.firstMention);
      to.firstMention = from.firstMention;
    }
    if (from.representative == to.representative){
    	System.out.println("!!!!!");
    	System.out.println(from.clusterID + " " +Arrays.toString(from.corefMentions.toArray()));
    	System.out.println(to.clusterID + " " + Arrays.toString(to.corefMentions.toArray()));
    	System.out.println(from.representative.corefClusterID + " -- " + to.representative.corefClusterID + " " + from.representative.mentionID + " " + to.representative.mentionID);
    }
    if(from.representative.moreRepresentativeThan(to.representative)) to.representative = from.representative;
    SieveCoreferenceSystem.logger.finer("merged clusters: "+toID+" += "+from.clusterID);
    to.printCorefCluster(SieveCoreferenceSystem.logger);
    from.printCorefCluster(SieveCoreferenceSystem.logger);
    SieveCoreferenceSystem.logger.finer("");
  }

  /** Print cluster information */
  public void printCorefCluster(Logger logger){
    logger.finer("Cluster ID: "+clusterID+"\tNumbers: "+numbers+"\tGenders: "+genders+"\tanimacies: "+animacies);
    logger.finer("NE: "+nerStrings+"\tfirst Mention's ID: "+firstMention.mentionID+"\tHeads: "+heads+"\twords: "+words);
    TreeMap<Integer, Mention> forSortedPrint = new TreeMap<>();
    for(Mention m : this.corefMentions){
      forSortedPrint.put(m.mentionID, m);
    }
    for(Mention m : forSortedPrint.values()){
      String rep = (representative == m)? "*":"";
      if(m.goldCorefClusterID==-1){
        logger.finer(rep + "mention-> id:"+m.mentionID+"\toriginalRef: "
                +m.originalRef+"\t"+m.spanToString() +"\tsentNum: "+m.sentNum+"\tstartIndex: "
                +m.startIndex+"\tType: "+m.mentionType+"\tNER: "+m.nerString);
      } else{
        logger.finer(rep + "mention-> id:"+m.mentionID+"\toriginalClusterID: "
                +m.goldCorefClusterID+"\t"+m.spanToString() +"\tsentNum: "+m.sentNum+"\tstartIndex: "
                +m.startIndex +"\toriginalRef: "+m.originalRef+"\tType: "+m.mentionType+"\tNER: "+m.nerString);
      }
    }
  }

  public boolean isSinglePronounCluster(Dictionaries dict){
    if(this.corefMentions.size() > 1) return false;
    for(Mention m : this.corefMentions) {
      if(m.isPronominal() || dict.allPronouns.contains(m.spanToString().toLowerCase())) return true;
    }
    return false;
  }

}
